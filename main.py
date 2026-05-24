import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(page_title="FellX4 — XRD Diffraktogramme", layout="wide")

st.title("🔬 FellX4 — XRD Diffraktogramm Viewer")


def parse_xy(content: str) -> tuple[list[float], list[float]] | None:
    """Parse .xy / .txt / .csv data into two_theta and intensity lists."""
    data = []
    for line in content.strip().splitlines():
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
        return None
    two_theta, intensity = zip(*data)
    return list(two_theta), list(intensity)


def plot_diffractogram(
    two_theta, intensity, title="Diffraktogramm", color="#1f77b4"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(two_theta, intensity, color=color, linewidth=1.0)
    ax.set_xlabel("2θ (°)", fontsize=11)
    ax.set_ylabel("Intensität (a.u.)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def interpolate_and_subtract(
    x1, y1, x2, y2
) -> tuple[list[float], list[float]]:
    """Interpolate (x2, y2) onto x1-grid, return (x1, y1 - y2_interp)."""
    y2_interp = np.interp(x1, x2, y2)
    diff = np.array(y1) - y2_interp
    return list(x1), diff.tolist()


# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(
    ["📁 Einzelnes Diffraktogramm", "📊 Vergleich + Subtraktion", "✏️ Manuelle Eingabe"]
)

# ==================== TAB 1: Einzeln ====================
with tab1:
    uploaded = st.file_uploader(
        "Wähle eine XRD-Datei (.xy, .txt, .csv)",
        type=["xy", "txt", "csv"],
        key="single",
    )

    if uploaded is not None:
        result = parse_xy(uploaded.read().decode("utf-8"))
        if result is None:
            st.error("Keine brauchbaren Daten gefunden.")
        else:
            two_theta, intensity = result
            fig = plot_diffractogram(
                two_theta, intensity, title=uploaded.name
            )
            st.pyplot(fig)

            with st.expander("📄 Rohdaten"):
                st.dataframe(
                    {"2θ (°)": two_theta, "Intensität": intensity},
                    use_container_width=True,
                )
    else:
        st.info("Lade eine Datei hoch, um das Diffraktogramm zu sehen.")

# ==================== TAB 2: Vergleich + Subtraktion ====================
with tab2:
    st.markdown(
        "Lade **zwei** Diffraktogramme — das zweite wird vom ersten **subtrahiert**."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        f_a = st.file_uploader(
            "Datei A (Minued)", type=["xy", "txt", "csv"], key="cmp_a"
        )
    with col_b:
        f_b = st.file_uploader(
            "Datei B (Subtrahend)", type=["xy", "txt", "csv"], key="cmp_b"
        )

    if f_a is not None and f_b is not None:
        data_a = parse_xy(f_a.read().decode("utf-8"))
        data_b = parse_xy(f_b.read().decode("utf-8"))

        if data_a is None or data_b is None:
            st.error("Eine der Dateien konnte nicht geparst werden.")
        else:
            x1, y1 = data_a
            x2, y2 = data_b

            # Interpolate onto the finer grid for best overlay
            if len(x1) >= len(x2):
                x_diff, y_diff = interpolate_and_subtract(x1, y1, x2, y2)
                x_overlay, y2_interp = x1, np.interp(x1, x2, y2)
                y1_label, y2_label = f"A ({f_a.name})", f"B ({f_b.name})"
            else:
                x_diff, y_diff = interpolate_and_subtract(x2, y2, x1, y1)
                x_overlay, y2_interp = x2, np.interp(x2, x1, y1)
                y1_label, y2_label = f"B ({f_b.name})", f"A ({f_a.name})"

            fig_cmp, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(10, 6), sharex=True,
                gridspec_kw={"height_ratios": [2, 1]},
            )

            # Upper: overlay
            ax1.plot(x1, y1, color="#1f77b4", linewidth=1.0, label=f"A: {f_a.name}")
            ax1.plot(x2, y2, color="#d62728", linewidth=1.0, label=f"B: {f_b.name}")
            ax1.set_ylabel("Intensität (a.u.)", fontsize=11)
            ax1.set_title("Überlagerung A + B", fontsize=12)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            # Lower: difference
            ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax2.plot(x_diff, y_diff, color="#2ca02c", linewidth=1.0)
            ax2.set_xlabel("2θ (°)", fontsize=11)
            ax2.set_ylabel("A − B (a.u.)", fontsize=11)
            ax2.set_title("Differenz A − B", fontsize=12)
            ax2.grid(True, alpha=0.3)

            fig_cmp.tight_layout()
            st.pyplot(fig_cmp)

            with st.expander("📄 Differenz-Rohdaten"):
                st.dataframe(
                    {"2θ (°)": x_diff, "A − B": y_diff},
                    use_container_width=True,
                )
    elif f_a is None and f_b is None:
        st.info("Lade beide Dateien hoch, um den Vergleich zu sehen.")

# ==================== TAB 3: Manuelle Eingabe ====================
with tab3:
    st.markdown(
        "Gib 2θ- und Intensitätswerte zeilenweise ein — getrennt durch Leerzeichen oder Tab."
    )
    manual = st.text_area(
        "Daten",
        height=200,
        placeholder="10.5 120\n12.3 450\n14.1 230\n...",
    )

    if st.button("Diagramm zeichnen") and manual.strip():
        result = parse_xy(manual)
        if result is None:
            st.error("Keine gültigen Zahlenpaare gefunden.")
        else:
            two_theta, intensity = result
            fig = plot_diffractogram(
                two_theta, intensity, title="Manuelles Diffraktogramm", color="#d62728"
            )
            st.pyplot(fig)

            with st.expander("📄 Rohdaten"):
                st.dataframe(
                    {"2θ (°)": two_theta, "Intensität": intensity},
                    use_container_width=True,
                )

st.caption("FellX4 — Version 2 | Vergleich + Subtraktion 🚀")
