import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re
import math

# ──────────────────────────────────────────────
#  Hilfsfunktionen
# ──────────────────────────────────────────────

def parse_xy(content: str):
    data = []
    for line in content.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                data.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not data:
        return None
    tt, intens = zip(*data)
    return list(tt), list(intens)


def plot_xy(tt, intens, title="Diffraktogramm", color="#1f77b4", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tt, intens, color=color, linewidth=1.0)
    ax.set_xlabel("2θ (°)", fontsize=11)
    ax.set_ylabel("Intensität (a.u.)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    fig = ax.figure
    fig.tight_layout()
    return fig


def parse_cif(content: str) -> dict | None:
    lines_raw = content.splitlines()
    lines = []
    multi_line = None
    for line in lines_raw:
        stripped = line.strip()
        if stripped.startswith("#") and multi_line is None:
            continue
        if multi_line is not None:
            multi_line.append(stripped)
            if stripped.endswith(";"):
                val = "\n".join(multi_line[1:-1])
                lines.append(val)
                multi_line = None
            continue
        if stripped.startswith(";"):
            multi_line = [stripped]
            continue
        lines.append(stripped)

    text = "\n".join(lines)
    cif = {}

    for key in ["_cell_length_a", "_cell_length_b", "_cell_length_c",
                "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"]:
        m = re.search(rf"{re.escape(key)}\s+([\d.eE+-]+(?:\(\d+\))?)\s*", text)
        if m:
            cif[key] = float(m.group(1).split("(")[0])
            continue
        m = re.search(rf"{re.escape(key)}\s*\n\s*([\d.eE+-]+(?:\(\d+\))?)", text)
        if m:
            cif[key] = float(m.group(1).split("(")[0])
            continue
        m = re.search(rf"{re.escape(key)}\s*=\s*([\d.eE+-]+(?:\(\d+\))?)", text)
        if m:
            cif[key] = float(m.group(1).split("(")[0])

    m = re.search(r"_symmetry_space_group_name_H-M\s+'([^']+)'", text)
    if m:
        cif["space_group"] = m.group(1)
    if "space_group" not in cif:
        m = re.search(r"_symmetry_space_group_name_H-M\s+([^\s]+)", text)
        if m:
            cif["space_group"] = m.group(1)

    # Atom sites
    atoms = []
    in_loop = False
    headers_raw = []
    values = []
    for line in lines_raw:
        ls = line.strip()
        if ls.startswith("#"):
            continue
        if ls == "loop_":
            in_loop = True
            headers_raw = []
            values = []
            continue
        if in_loop:
            if ls.startswith("_atom_site.") or ls.startswith("_atom_site_"):
                headers_raw.append(ls)
            elif headers_raw and ls:
                if ls.startswith("_"):
                    in_loop = False
                else:
                    values.append(ls.split())
            elif not headers_raw and ls.startswith("_"):
                in_loop = False

    if headers_raw and values:
        prefix = "_atom_site." if headers_raw[0].startswith("_atom_site.") else "_atom_site_"
        header_keys = [h[len(prefix):] for h in headers_raw]
        for row in values:
            if len(row) >= len(header_keys):
                atom = dict(zip(header_keys, row))
                atoms.append(atom)

    if not atoms:
        atom_data = {}
        current_key = None
        for line in lines_raw:
            ls = line.strip()
            if ls.startswith("#"):
                continue
            if ls.startswith("_atom_site.") or ls.startswith("_atom_site_"):
                parts = ls.split(None, 1)
                if len(parts) >= 2:
                    key = parts[0]
                    val = parts[1]
                    prefix = "_atom_site." if key.startswith("_atom_site.") else "_atom_site_"
                    short_key = key[len(prefix):]
                    if short_key not in atom_data:
                        atom_data[short_key] = []
                    atom_data[short_key].append(val)
                else:
                    current_key = ls
            elif current_key and ls and not ls.startswith("_"):
                prefix = "_atom_site." if current_key.startswith("_atom_site.") else "_atom_site_"
                short_key = current_key[len(prefix):]
                if short_key not in atom_data:
                    atom_data[short_key] = []
                atom_data[short_key].append(ls)
                current_key = None

        if atom_data:
            keys = list(atom_data.keys())
            n_rows = max(len(v) for v in atom_data.values())
            for i in range(n_rows):
                atom = {}
                for k in keys:
                    if i < len(atom_data[k]):
                        atom[k] = atom_data[k][i]
                if atom:
                    atoms.append(atom)

    if atoms:
        cif["atoms"] = atoms

    return cif if any(k in cif for k in ["_cell_length_a", "atoms"]) else None


# ──────────────────────────────────────────────
#  HOME — Upload-Seite
# ──────────────────────────────────────────────

st.title("🔬 FellX4 — XRD Toolkit")
st.markdown("Lade CIF + Diffraktogramm hoch, dann geht's zur Analyse.")

col_cif, col_xrd = st.columns(2)

with col_cif:
    st.markdown("### 📄 CIF-Datei")
    cif_file = st.file_uploader("Kristallstruktur (.cif)", type=["cif"], key="home_cif")
    if cif_file:
        crystal = parse_cif(cif_file.read().decode("utf-8"))
        if crystal and "atoms" in crystal and "_cell_length_a" in crystal:
            a = crystal["_cell_length_a"]
            b = crystal["_cell_length_b"]
            c = crystal["_cell_length_c"]
            alpha = crystal["_cell_angle_alpha"]
            beta = crystal["_cell_angle_beta"]
            gamma = crystal["_cell_angle_gamma"]
            sg = crystal.get("space_group", "—")

            st.success(
                f"✅ **CIF geladen** — "
                f"a={a:.3f} b={b:.3f} c={c:.3f} Å, "
                f"α={alpha:.2f} β={beta:.2f} γ={gamma:.2f}°, "
                f"{sg} | {len(crystal['atoms'])} Atome"
            )
            with st.expander("📄 Atompositionen"):
                st.dataframe(crystal["atoms"], use_container_width=True)

            st.session_state["crystal"] = crystal
            st.session_state["a"] = a
            st.session_state["b"] = b
            st.session_state["c"] = c
            st.session_state["alpha"] = alpha
            st.session_state["beta"] = beta
            st.session_state["gamma"] = gamma
            st.session_state["sg"] = sg
            st.session_state["cif_file_name"] = cif_file.name
        else:
            st.error("CIF konnte nicht vollständig geparst werden — brauche _cell_length_* und _atom_site_* Einträge.")
            st.session_state.pop("crystal", None)

with col_xrd:
    st.markdown("### 📊 Diffraktogramm")
    xrd_file = st.file_uploader("XRD-Datei (.xy, .txt, .csv)", type=["xy", "txt", "csv"], key="home_xrd")
    if xrd_file:
        xrd = parse_xy(xrd_file.read().decode("utf-8"))
        if xrd:
            tt, intens = xrd
            st.pyplot(plot_xy(tt, intens, title=xrd_file.name))
            st.info(f"{len(tt)} Datenpunkte")

            prominence = st.slider("Peak-Empfindlichkeit", 0.0, 0.5, 0.05, 0.01,
                                   key="home_prom", help="Niedriger = mehr Peaks")

            from scipy.signal import find_peaks as sp_find_peaks
            from scipy.ndimage import gaussian_filter1d
            arr = np.array(intens, dtype=float)
            if np.max(arr) > 0:
                smoothed = gaussian_filter1d(arr, sigma=1.5)
                abs_prom = prominence * np.max(arr)
                step = np.median(np.diff(tt)) if len(tt) > 1 else 0.1
                distance = max(3, int(0.5 / step))
                peaks_idx, _ = sp_find_peaks(smoothed, prominence=abs_prom, distance=distance, width=1)
                peaks = [(tt[i], intens[i]) for i in peaks_idx]
                st.info(f"{len(peaks)} Peaks detektiert")
                st.session_state["peaks"] = peaks
            else:
                st.session_state["peaks"] = []

            st.session_state["tt_raw"] = tt
            st.session_state["intens_raw"] = intens
            st.session_state["xrd_file_name"] = xrd_file.name
        else:
            st.error("Konnte Diffraktogramm nicht parsen.")

# Wavelength
st.markdown("---")
wavelength = st.number_input("Wellenlänge λ (Å)", value=1.5406, format="%.4f",
                              help="Cu Kα = 1.5406, Mo Kα = 0.7107")
st.session_state["wavelength"] = wavelength

# Start-Button
st.markdown("---")
col_btn, _ = st.columns([1, 3])
with col_btn:
    can_start = "crystal" in st.session_state and "tt_raw" in st.session_state
    if st.button("🚀 Analyse starten", type="primary", use_container_width=True, disabled=not can_start):
        st.switch_page("pages/1_Analyse.py")

if not can_start:
    st.info("Lade beide Dateien hoch (CIF + Diffraktogramm), um zur Analyse zu gelangen.")
