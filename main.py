import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(page_title="FellX4 — XRD Diffraktogramme", layout="wide")

st.title("🔬 FellX4 — XRD Diffraktogramm Viewer")

st.markdown(
    "Lade eine `.xy`-Datei hoch oder gib manuell 2θ-Werte und Intensitäten ein."
)

tab1, tab2 = st.tabs(["📁 Datei laden", "✏️ Manuelle Eingabe"])

# ---------- TAB 1: Datei Upload ----------
with tab1:
    uploaded = st.file_uploader(
        "Wähle eine XRD-Datei (.xy, .txt, .csv)", type=["xy", "txt", "csv"]
    )

    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        data = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    data.append((x, y))
                except ValueError:
                    continue

        if not data:
            st.error("Keine brauchbaren Daten gefunden.")
        else:
            two_theta, intensity = zip(*data)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(two_theta, intensity, color="#1f77b4", linewidth=1.0)
            ax.set_xlabel("2θ (°)", fontsize=11)
            ax.set_ylabel("Intensität (a.u.)", fontsize=11)
            ax.set_title(uploaded.name, fontsize=12)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

            with st.expander("📄 Rohdaten"):
                st.dataframe(
                    {"2θ (°)": two_theta, "Intensität": intensity},
                    use_container_width=True,
                )
    else:
        st.info("Lade eine Datei hoch, um das Diffraktogramm zu sehen.")

# ---------- TAB 2: Manuelle Eingabe ----------
with tab2:
    st.markdown("Gib 2θ- und Intensitätswerte zeilenweise ein — getrennt durch Leerzeichen oder Tab.")
    manual = st.text_area("Daten", height=200, placeholder="10.5 120\n12.3 450\n14.1 230\n...")

    if st.button("Diagramm zeichnen") and manual.strip():
        data = []
        for line in manual.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    data.append((x, y))
                except ValueError:
                    continue

        if not data:
            st.error("Keine gültigen Zahlenpaare gefunden.")
        else:
            two_theta, intensity = zip(*data)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(two_theta, intensity, color="#d62728", linewidth=1.0)
            ax.set_xlabel("2θ (°)", fontsize=11)
            ax.set_ylabel("Intensität (a.u.)", fontsize=11)
            ax.set_title("Manuelles Diffraktogramm", fontsize=12)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

            with st.expander("📄 Rohdaten"):
                st.dataframe(
                    {"2θ (°)": two_theta, "Intensität": intensity},
                    use_container_width=True,
                )

st.caption("FellX4 — erster Wurf 🚀")
