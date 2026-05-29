import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re
import math

st.set_page_config(page_title="FellX4 — XRD Toolkit", layout="wide")

st.title("🔬 FellX4 — XRD Diffraktogramme & Strukturfaktoren")

# ──────────────────────────────────────────────
#  Hilfsfunktionen
# ──────────────────────────────────────────────

def parse_xy(content: str):
    """Parse .xy / .txt / .csv data into two_theta and intensity lists."""
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


def find_peaks(tt, intens, prominence=0.05):
    """Peak detection with scipy.signal.find_peaks.
    prominence = minimal relative prominence (fraction of max intensity).
    Returns list of (2theta, intensity) tuples."""
    from scipy.signal import find_peaks as sp_find_peaks
    from scipy.ndimage import gaussian_filter1d

    arr = np.array(intens, dtype=float)
    # Sanity: flat/empty data
    if np.max(arr) == 0:
        return []

    # Gaussian smoothing (sigma=1.5 px) to reduce noise before peak detection
    smoothed = gaussian_filter1d(arr, sigma=1.5)

    # Absolute prominence threshold
    abs_prominence = prominence * np.max(arr)

    # Minimum distance between peaks (in data points, ~0.5° 2θ)
    # Use median step size as reference
    if len(tt) > 1:
        step = np.median(np.diff(tt))
        distance = max(3, int(0.5 / step))  # ~0.5° 2θ minimum separation
    else:
        distance = 3

    peaks_idx, props = sp_find_peaks(
        smoothed,
        prominence=abs_prominence,
        distance=distance,
        width=1,  # at least 1 data point wide
    )

    return [(tt[i], intens[i]) for i in peaks_idx]


# ──────────────────────────────────────────────
#  CIF-Parser
# ──────────────────────────────────────────────

def parse_cif(content: str) -> dict | None:
    """Parse basic CIF: unit cell, space group, atom sites.
    Supports both _atom_site.xxx and _atom_site_xxx notations.
    Handles values on same line, next line, or semicolon-delimited."""

    lines_raw = content.splitlines()
    # Remove comments and strip
    lines = []
    multi_line = None  # track ;...; blocks
    for line in lines_raw:
        stripped = line.strip()
        if stripped.startswith("#") and multi_line is None:
            continue
        if multi_line is not None:
            multi_line.append(stripped)
            if stripped.endswith(";"):
                # End of semicolon block
                val = "\n".join(multi_line[1:-1])  # remove opening and closing ;
                lines.append(val)
                multi_line = None
            continue
        if stripped.startswith(";"):
            multi_line = [stripped]
            continue
        lines.append(stripped)

    text = "\n".join(lines)

    cif = {}

    # --- Unit cell (supports same line, next line, = separator) ---
    for key in ["_cell_length_a", "_cell_length_b", "_cell_length_c",
                "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"]:
        # Same line: key value
        m = re.search(rf"{re.escape(key)}\s+([\d.eE+-]+(?:\(\d+\))?)\s*", text)
        if m:
            cif[key] = float(m.group(1).split("(")[0])
            continue
        # Next line: key\nvalue
        m = re.search(rf"{re.escape(key)}\s*\n\s*([\d.eE+-]+(?:\(\d+\))?)", text)
        if m:
            cif[key] = float(m.group(1).split("(")[0])
            continue
        # Key=value
        m = re.search(rf"{re.escape(key)}\s*=\s*([\d.eE+-]+(?:\(\d+\))?)", text)
        if m:
            cif[key] = float(m.group(1).split("(")[0])

    # --- Space group ---
    m = re.search(r"_symmetry_space_group_name_H-M\s+'([^']+)'", text)
    if m:
        cif["space_group"] = m.group(1)
    if "space_group" not in cif:
        m = re.search(r"_symmetry_space_group_name_H-M\s+([^\s]+)", text)
        if m:
            cif["space_group"] = m.group(1)

    # --- Atom sites ---
    atoms = []
    seen_keys = set()

    # Strategy 1: Parse loop_ blocks
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

    # Strategy 2: Parse non-loop _atom_site_ entries
    if not atoms:
        # Gather all _atom_site_ keys and values
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
            # Convert column-based to row-based
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


def d_spacing(h, k, l, a, b, c, alpha, beta, gamma):
    """Calculate d-spacing for given hkl in any crystal system."""
    # Convert to radians
    al = math.radians(alpha)
    be = math.radians(beta)
    ga = math.radians(gamma)

    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    sa, sb, sg = math.sin(al), math.sin(be), math.sin(ga)

    # Volume of reciprocal cell
    vol = a * b * c * math.sqrt(
        1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
    )

    # Reciprocal metric tensor (simplified for general case)
    a_star = b * c * sa / vol
    b_star = a * c * sb / vol
    c_star = a * b * sg / vol

    ca_star = (cb * cg - ca) / (sb * sg)
    cb_star = (ca * cg - cb) / (sa * sg)
    cg_star = (ca * cb - cg) / (sa * sb)

    d2 = (
        h**2 * a_star**2
        + k**2 * b_star**2
        + l**2 * c_star**2
        + 2 * h * k * a_star * b_star * cg_star
        + 2 * h * l * a_star * c_star * cb_star
        + 2 * k * l * b_star * c_star * ca_star
    )
    if d2 <= 0:
        return None
    return 1.0 / math.sqrt(d2)


# ──────────────────────────────────────────────
#  Atomformfaktoren (analytisch, 4 Gauß + c)
#  Quelle: International Tables Vol. C, Waasmaier & Kirfel 1995
# ──────────────────────────────────────────────

# Format: {Element: [(a1,b1), (a2,b2), (a3,b3), (a4,b4), c]}
SCATTERING_FACTORS = {
    "H":  [(0.493, 10.511), (0.323, 26.126), (0.140, 3.142), (0.041, 57.800), 0.003],
    "C":  [(2.310, 20.844), (1.020, 10.208), (1.589, 0.569), (0.865, 51.651), 0.216],
    "N":  [(12.213, 0.006), (3.132, 9.893), (2.013, 28.665), (1.166, 0.396), -11.529],
    "O":  [(3.049, 13.277), (2.287, 5.701), (1.546, 0.324), (0.867, 32.909), 0.251],
    "Na": [(4.763, 3.285), (3.174, 8.842), (1.268, 0.314), (1.113, 129.424), 0.676],
    "Mg": [(5.420, 2.828), (2.174, 79.261), (1.227, 0.381), (0.859, 21.806), 0.318],
    "Al": [(6.420, 3.039), (1.594, 77.558), (1.465, 0.402), (1.043, 21.024), 0.477],
    "Si": [(6.292, 2.439), (3.035, 32.334), (1.989, 0.678), (1.541, 81.694), 0.145],
    "P":  [(6.435, 1.907), (4.179, 27.157), (1.781, 0.647), (1.165, 67.913), 0.442],
    "S":  [(6.905, 1.468), (5.203, 22.215), (1.438, 0.254), (1.586, 55.925), 0.867],
    "Cl": [(11.460, 0.010), (7.196, 1.166), (6.256, 18.520), (1.646, 47.778), -9.557],
    "K":  [(8.219, 12.795), (7.440, 0.775), (1.052, 213.719), (0.866, 41.684), 0.424],
    "Ca": [(8.627, 10.442), (7.387, 0.660), (1.590, 85.748), (1.021, 178.438), 0.375],
    "Ti": [(9.759, 7.851), (7.356, 0.472), (1.699, 37.267), (1.203, 111.638), 0.982],
    "Fe": [(11.776, 1.035), (7.122, 11.441), (4.148, 0.656), (2.400, 53.144), 0.557],
    "Ni": [(12.838, 1.503), (7.292, 11.395), (4.284, 0.462), (2.255, 50.719), 0.333],
    "Cu": [(13.338, 3.583), (7.168, 0.231), (5.616, 14.079), (2.263, 49.259), 0.616],
    "Zn": [(14.074, 3.265), (7.032, 0.233), (5.165, 12.895), (2.411, 44.571), 1.319],
    "Sr": [(19.215, 17.462), (16.360, 2.661), (4.077, 71.450), (2.361, 0.013), 0.981],
    "Ba": [(24.307, 0.002), (17.635, 16.115), (6.992, 0.430), (4.350, 70.900), 0.718],
    "Pb": [(32.366, 0.033), (22.675, 10.241), (9.404, 0.513), (5.462, 52.131), 4.095],
}


def atomic_scattering_factor(element: str, sin_theta_over_lambda: float) -> float:
    """Calculate f(sinθ/λ) using 4-Gaussian approximation."""
    coeffs = SCATTERING_FACTORS.get(element.capitalize())
    if coeffs is None:
        return 0.0  # Unknown element — skip contribution
    s2 = sin_theta_over_lambda**2
    f = coeffs[4]  # c
    for a_i, b_i in coeffs[:4]:
        f += a_i * math.exp(-b_i * s2)
    return f


def compute_structure_factors(
    atoms: list[dict],
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
    wavelength: float,
    hkl_ranges: tuple[int, int, int] = (10, 10, 10),
    min_f_sq_frac: float = 0.01,  # nur Reflexe >1% des max |F|²
):
    """Compute F(hkl) for all hkl in range.
    min_f_sq_frac: fraction of max |F|² below which reflections are dropped."""
    h_max, k_max, l_max = hkl_ranges
    raw_results = []
    for h in range(-h_max, h_max + 1):
        for k in range(-k_max, k_max + 1):
            for l in range(-l_max, l_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                d = d_spacing(h, k, l, a, b, c, alpha, beta, gamma)
                if d is None or d <= 0:
                    continue
                sintheta = wavelength / (2 * d)
                if abs(sintheta) > 1:
                    continue
                theta = math.degrees(math.asin(sintheta))
                two_theta = 2 * theta
                if two_theta < 5 or two_theta > 150:
                    continue
                sin_t_over_l = math.sin(math.radians(theta)) / wavelength

                # Sum over atoms
                F_real = 0.0
                F_imag = 0.0
                for atom in atoms:
                    try:
                        x = float(atom.get("fract_x", 0))
                        y = float(atom.get("fract_y", 0))
                        z = float(atom.get("fract_z", 0))
                        occ = float(atom.get("occupancy", 1.0))
                        match = re.match(r"([A-Za-z]+)", atom.get("type_symbol", "H"))
                        element = match.group(1) if match else "H"
                    except (ValueError, AttributeError):
                        continue
                    f_j = atomic_scattering_factor(element, sin_t_over_l)
                    phase = 2 * math.pi * (h * x + k * y + l * z)
                    F_real += occ * f_j * math.cos(phase)
                    F_imag += occ * f_j * math.sin(phase)

                F_sq = F_real**2 + F_imag**2
                if F_sq > 0.001:
                    raw_results.append({
                        "h": h, "k": k, "l": l,
                        "d (Å)": round(d, 4),
                        "2θ (°)": round(two_theta, 4),
                        "|F|²": round(F_sq, 2),
                        "|F|": round(math.sqrt(F_sq), 2),
                    })

    # Filter: nur Reflexe > min_f_sq_frac * max |F|²
    if raw_results:
        max_f = max(r["|F|²"] for r in raw_results)
        threshold = min_f_sq_frac * max_f
        results = [r for r in raw_results if r["|F|²"] >= threshold]
    else:
        results = []

    results.sort(key=lambda r: r["d (Å)"], reverse=True)
    return results


def match_peaks_to_hkl(
    peaks: list[tuple[float, float]],
    hkl_data: list[dict],
    wavelength: float,
    tol: float = 0.2,
) -> list[dict]:
    """Match experimental 2θ peaks to calculated hkl reflections."""
    matched = []
    for tt_obs, intens in peaks:
        best = None
        best_delta = float("inf")
        for ref in hkl_data:
            delta = abs(ref["2θ (°)"] - tt_obs)
            if delta < best_delta and delta < tol:
                best_delta = delta
                best = ref
        matched.append({
            "2θ obs": round(tt_obs, 4),
            "Intensität": round(intens, 2),
            "h": best["h"] if best else "—",
            "k": best["k"] if best else "—",
            "l": best["l"] if best else "—",
            "d (Å)": best["d (Å)"] if best else "—",
            "Δ2θ": round(best_delta, 4) if best else "—",
            "|F|": best["|F|"] if best else "—",
        })
    return matched


# ──────────────────────────────────────────────
#  TABS
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Diffraktogramm",
    "📊 Vergleich + Subtraktion",
    "🔷 HKL + Strukturfaktoren",
    "✏️ Manuelle Eingabe",
])

# ==================== TAB 1 ====================
with tab1:
    uploaded = st.file_uploader("XRD-Datei (.xy, .txt, .csv)", type=["xy", "txt", "csv"], key="s1")
    if uploaded:
        res = parse_xy(uploaded.read().decode("utf-8"))
        if res:
            tt, intens = res
            st.pyplot(plot_xy(tt, intens, title=uploaded.name))
            with st.expander("📄 Rohdaten"):
                st.dataframe({"2θ (°)": tt, "Intensität": intens}, use_container_width=True)
        else:
            st.error("Keine brauchbaren Daten.")
    else:
        st.info("Lade eine Datei hoch.")

# ==================== TAB 2 ====================
with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        fa = st.file_uploader("Datei A", type=["xy", "txt", "csv"], key="c2a")
    with col_b:
        fb = st.file_uploader("Datei B", type=["xy", "txt", "csv"], key="c2b")

    if fa and fb:
        da = parse_xy(fa.read().decode("utf-8"))
        db = parse_xy(fb.read().decode("utf-8"))
        if da and db:
            x1, y1 = da
            x2, y2 = db
            if len(x1) >= len(x2):
                y2_interp = np.interp(x1, x2, y2)
                diff = np.array(y1) - y2_interp
                x_diff = x1
            else:
                y1_interp = np.interp(x2, x1, y1)
                diff = np.array(y1_interp) - y2
                x_diff = x2

            fig_cmp, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                                gridspec_kw={"height_ratios": [2, 1]})
            ax1.plot(x1, y1, color="#1f77b4", lw=1.0, label=f"A: {fa.name}")
            ax1.plot(x2, y2, color="#d62728", lw=1.0, label=f"B: {fb.name}")
            ax1.set_ylabel("Intensität (a.u.)", fontsize=11)
            ax1.set_title("Überlagerung A + B", fontsize=12)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax2.axhline(0, color="gray", lw=0.8, ls="--")
            ax2.plot(x_diff, diff, color="#2ca02c", lw=1.0)
            ax2.set_xlabel("2θ (°)", fontsize=11)
            ax2.set_ylabel("A − B (a.u.)", fontsize=11)
            ax2.set_title("Differenz A − B", fontsize=12)
            ax2.grid(True, alpha=0.3)
            fig_cmp.tight_layout()
            st.pyplot(fig_cmp)
    elif not fa and not fb:
        st.info("Lade beide Dateien für den Vergleich.")

# ==================== TAB 3: HKL + Strukturfaktoren ====================
with tab3:
    st.markdown("### 📄 CIF-Datei → Kristallstruktur")

    cif_file = st.file_uploader(
        "CIF hochladen (unit cell + atom positions)", type=["cif"], key="cif"
    )

    wavelength = st.number_input(
        "Wellenlänge λ (Å)", value=1.5406, format="%.4f",
        help="Cu Kα = 1.5406 Å, Mo Kα = 0.7107 Å"
    )

    hkl_range = st.slider("hkl-Suchbereich", 1, 10, 5, help="Maximaler Index für h, k, l")

    crystal = None
    if cif_file is not None:
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
                f"✅ CIF geladen: "
                f"a={a:.3f} b={b:.3f} c={c:.3f} Å, "
                f"α={alpha:.2f} β={beta:.2f} γ={gamma:.2f}°, "
                f"Raumgruppe: {sg} | "
                f"{len(crystal['atoms'])} Atome"
            )
            with st.expander("📄 Atompositionen"):
                st.dataframe(crystal["atoms"], use_container_width=True)
        else:
            st.error("CIF konnte nicht vollständig geparst werden — brauche _cell_length_* und _atom_site_* Einträge.")

    st.markdown("---")
    st.markdown("### 📊 Diffraktogramm (experimentell)")

    xrd_file = st.file_uploader(
        "Diffraktogramm (.xy)", type=["xy", "txt", "csv"], key="xrd4hkl"
    )

    prominence = st.slider(
        "Peak-Empfindlichkeit", 0.0, 0.5, 0.05, 0.01,
        help="Niedriger = mehr Peaks erkannt"
    )

    peaks = []
    if xrd_file is not None:
        xrd = parse_xy(xrd_file.read().decode("utf-8"))
        if xrd:
            tt, intens = xrd
            peaks = find_peaks(tt, intens, prominence=prominence)

            fig_p, (ax_p, ax_peaks) = plt.subplots(
                2, 1, figsize=(10, 5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax_p.plot(tt, intens, color="#1f77b4", lw=1.0)
            peak_tt = [p[0] for p in peaks]
            peak_int = [p[1] for p in peaks]
            ax_p.scatter(peak_tt, peak_int, color="red", s=30, zorder=5, label=f"{len(peaks)} Peaks")
            ax_p.set_ylabel("Intensität", fontsize=11)
            ax_p.set_title(xrd_file.name, fontsize=12)
            ax_p.legend(fontsize=9)
            ax_p.grid(True, alpha=0.3)

            ax_peaks.bar(peak_tt, peak_int, width=0.08, color="red", alpha=0.6)
            ax_peaks.set_xlabel("2θ (°)", fontsize=11)
            ax_peaks.set_ylabel("Peak-Intensität", fontsize=11)
            ax_peaks.grid(True, alpha=0.3)
            fig_p.tight_layout()
            st.pyplot(fig_p)

            st.info(f"{len(peaks)} Peaks detektiert")
        else:
            st.error("Diffraktogramm konnte nicht geparst werden.")

    st.markdown("---")

    if st.button("🧮 hkl-Indizierung + Strukturfaktoren berechnen", type="primary"):
        if crystal is None or "_cell_length_a" not in crystal or "atoms" not in crystal:
            st.error("❌ Bitte zuerst eine gültige CIF-Datei hochladen.")
        elif not peaks:
            st.error("❌ Keine Peaks im Diffraktogramm — Datei hochladen oder Empfindlichkeit erhöhen.")
        else:
            a = crystal["_cell_length_a"]
            b = crystal["_cell_length_b"]
            c = crystal["_cell_length_c"]
            alpha = crystal["_cell_angle_alpha"]
            beta = crystal["_cell_angle_beta"]
            gamma = crystal["_cell_angle_gamma"]

            with st.spinner("Berechne hkl-Reflexe..."):
                hkl_refs = compute_structure_factors(
                    crystal["atoms"], a, b, c, alpha, beta, gamma,
                    wavelength, (hkl_range, hkl_range, hkl_range),
                )

            st.success(
                f"✅ {len(hkl_refs)} theoretische Reflexe im Bereich 5° ≤ 2θ ≤ 150° berechnet."
            )

            with st.expander("📋 Alle berechneten Reflexe"):
                st.dataframe(hkl_refs, use_container_width=True, hide_index=True)

            # Match peaks to hkl
            matched = match_peaks_to_hkl(peaks, hkl_refs, wavelength, tol=0.5)

            st.markdown("### 🎯 Peak-Zuordnung (hkl)")
            st.dataframe(matched, use_container_width=True, hide_index=True)

            matched_count = sum(1 for m in matched if m["h"] != "—")
            st.info(f"{matched_count} von {len(matched)} Peaks konnten hkl zugeordnet werden.")

            # Plot overlay
            if hkl_refs:
                fig_overlay, ax_o = plt.subplots(figsize=(10, 4))
                # Experimental
                if peaks:
                    p_tt = [p[0] for p in peaks]
                    p_int = [p[1] for p in peaks]
                    # Normalize intensity to max ref |F|² for visual comparison
                    if p_int:
                        p_int_norm = np.array(p_int) / max(p_int) * 100
                        ax_o.bar(p_tt, p_int_norm, width=0.08, color="red", alpha=0.5, label="Experiment")

                # Calculated reflections (stick diagram)
                ref_tt = [r["2θ (°)"] for r in hkl_refs]
                ref_f = [r["|F|²"] for r in hkl_refs]
                if ref_f:
                    ref_f_norm = np.array(ref_f) / max(ref_f) * 100
                    ax_o.vlines(ref_tt, 0, ref_f_norm, color="#1f77b4", lw=1.5, alpha=0.8, label="Berechnet (|F|²)")

                ax_o.set_xlabel("2θ (°)", fontsize=11)
                ax_o.set_ylabel("Norm. Intensität / |F|²", fontsize=11)
                ax_o.set_title("Experiment vs. Berechnet (Stick-Diagramm)", fontsize=12)
                ax_o.legend(fontsize=9)
                ax_o.grid(True, alpha=0.3)
                fig_overlay.tight_layout()
                st.pyplot(fig_overlay)

# ==================== TAB 4 ====================
with tab4:
    st.markdown("2θ- und Intensitätswerte zeilenweise — Leerzeichen oder Tab getrennt.")
    manual = st.text_area("Daten", height=200, placeholder="10.5 120\n12.3 450\n14.1 230\n...")
    if st.button("Diagramm zeichnen") and manual.strip():
        res = parse_xy(manual)
        if res:
            tt, intens = res
            st.pyplot(plot_xy(tt, intens, title="Manuelles Diffraktogramm", color="#d62728"))
            with st.expander("📄 Rohdaten"):
                st.dataframe({"2θ (°)": tt, "Intensität": intens}, use_container_width=True)
        else:
            st.error("Keine gültigen Zahlenpaare.")

st.caption("FellX4 — mit HKL-Suche & Strukturfaktoren 🚀🔷")
