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
                    phase_rad = math.atan2(F_imag, F_real)
                    raw_results.append({
                        "h": h, "k": k, "l": l,
                        "d (Å)": round(d, 4),
                        "2θ (°)": round(two_theta, 4),
                        "|F|²": round(F_sq, 2),
                        "|F|": round(math.sqrt(F_sq), 2),
                        "F_real": round(F_real, 3),
                        "F_imag": round(F_imag, 3),
                        "φ (°)": round(math.degrees(phase_rad), 1),
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
#  3D-Elementarzelle
# ──────────────────────────────────────────────

def frac_to_cart(frac, a, b, c, alpha, beta, gamma):
    """Convert fractional (x,y,z) to Cartesian coordinates.
    Standard orientation: a∥x, b in xy-plane."""
    ca = math.cos(math.radians(alpha))
    cb = math.cos(math.radians(beta))
    cg = math.cos(math.radians(gamma))
    sg = math.sin(math.radians(gamma))

    # Basis vectors
    ax = a
    bx = b * cg
    by = b * sg
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz = c * math.sqrt(1 - cb**2 - ((ca - cb * cg) / sg)**2)

    x, y, z = frac
    return (
        x * ax + y * bx + z * cx,
        y * by + z * cy,
        z * cz,
    )


def plot_unit_cell(atoms, a, b, c, alpha, beta, gamma):
    """Matplotlib 3D plot of the unit cell with atoms."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # 8 corners of the unit cell in fractional coordinates
    corners_frac = [
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1),
    ]
    corners = [frac_to_cart(corner, a, b, c, alpha, beta, gamma)
               for corner in corners_frac]

    def edges_from_corners(c):
        """Return 12 edges of a hexahedron."""
        idx = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        return [(c[i], c[j]) for i, j in idx]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Draw edges
    for p1, p2 in edges_from_corners(corners):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color="#1f77b4", lw=1.5)

    # Shade cell faces (slightly transparent)
    faces_idx = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)]
    face_verts = [[corners[i] for i in f] for f in faces_idx]
    poly = Poly3DCollection(face_verts, alpha=0.05, facecolor="#1f77b4",
                            edgecolor="none")
    ax.add_collection3d(poly)

    # Plot atoms
    colors = {
        "H": "#ffffff", "C": "#222222", "N": "#3050f8", "O": "#ff0d0d",
        "Na": "#ab5cf2", "Mg": "#8aff00", "Al": "#bfa6a6", "Si": "#f0c8a0",
        "P": "#ff8000", "S": "#ffff30", "Cl": "#1ff01f", "K": "#8f40d4",
        "Ca": "#3dff00", "Ti": "#bfc2c7", "Fe": "#e06633", "Ni": "#50f050",
        "Cu": "#c88033", "Zn": "#71d0a0", "Sr": "#00ff00", "Ba": "#00c900",
        "Pb": "#575961",
    }

    for atom in atoms:
        try:
            fx = float(atom.get("fract_x", 0))
            fy = float(atom.get("fract_y", 0))
            fz = float(atom.get("fract_z", 0))
            occ = float(atom.get("occupancy", 1.0))
            match = re.match(r"([A-Za-z]+)", atom.get("type_symbol", "H"))
            element = match.group(1) if match else "H"
        except (ValueError, AttributeError):
            continue

        pos = frac_to_cart((fx, fy, fz), a, b, c, alpha, beta, gamma)
        el_color = colors.get(element.capitalize(), "#888888")
        size = 80 + 40 * (occ if occ <= 1 else 1)
        ax.scatter(*pos, c=el_color, s=size, edgecolors="black",
                   linewidths=0.3, alpha=0.85, zorder=10)

    # Axis labels
    a_vec = frac_to_cart((1.15, 0, 0), a, b, c, alpha, beta, gamma)
    b_vec = frac_to_cart((0, 1.15, 0), a, b, c, alpha, beta, gamma)
    c_vec = frac_to_cart((0, 0, 1.15), a, b, c, alpha, beta, gamma)
    ax.text(*a_vec, "a", fontsize=12, fontweight="bold", color="#1f77b4")
    ax.text(*b_vec, "b", fontsize=12, fontweight="bold", color="#1f77b4")
    ax.text(*c_vec, "c", fontsize=12, fontweight="bold", color="#1f77b4")

    # Equal aspect
    all_pts = np.array(corners)
    mx = np.max(np.abs(all_pts))
    ax.set_xlim(-mx*0.1, mx*1.1)
    ax.set_ylim(-mx*0.1, mx*1.1)
    ax.set_zlim(-mx*0.1, mx*1.1)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title("Elementarzelle", fontsize=12, fontweight="bold")

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
#  TABS
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Diffraktogramm",
    "📊 Vergleich + Subtraktion",
    "🔷 HKL + Strukturfaktoren",
    "✏️ Manuelle Eingabe",
    "📐 FWHM + Scherrer",
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
    tol = st.slider("Toleranz Δ2θ (°)", 0.05, 1.0, 0.3, 0.05,
                    help="Maximale Abweichung für hkl-Zuordnung")

    crystal = None
    cif_ok = False
    a = b = c = alpha = beta = gamma = 0.0
    sg = "—"
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
            cif_ok = True

            st.success(
                f"✅ CIF geladen: "
                f"a={a:.3f} b={b:.3f} c={c:.3f} Å, "
                f"α={alpha:.2f} β={beta:.2f} γ={gamma:.2f}°, "
                f"Raumgruppe: {sg} | "
                f"{len(crystal['atoms'])} Atome"
            )
            with st.expander("📄 Atompositionen"):
                st.dataframe(crystal["atoms"], use_container_width=True)

            if st.checkbox("🔬 3D-Elementarzelle anzeigen", value=True):
                fig_cell = plot_unit_cell(
                    crystal["atoms"], a, b, c, alpha, beta, gamma
                )
                st.pyplot(fig_cell)
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
    tt_raw, intens_raw = None, None
    if xrd_file is not None:
        xrd = parse_xy(xrd_file.read().decode("utf-8"))
        if xrd:
            tt_raw, intens_raw = xrd
            peaks = find_peaks(tt_raw, intens_raw, prominence=prominence)

            fig_p, (ax_p, ax_peaks) = plt.subplots(
                2, 1, figsize=(10, 5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax_p.plot(tt_raw, intens_raw, color="#1f77b4", lw=1.0)
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

    # ── Berechnung ────────────────────────────────────────────
    if st.button("🧮 hkl-Indizierung + Strukturfaktoren berechnen", type="primary"):
        if crystal is None or not cif_ok:
            st.error("❌ Bitte zuerst eine gültige CIF-Datei hochladen.")
        elif not peaks:
            st.error("❌ Keine Peaks im Diffraktogramm — Datei hochladen oder Empfindlichkeit erhöhen.")
        else:
            with st.spinner("Berechne hkl-Reflexe..."):
                hkl_refs = compute_structure_factors(
                    crystal["atoms"], a, b, c, alpha, beta, gamma,
                    wavelength, (hkl_range, hkl_range, hkl_range),
                )

            st.success(f"✅ {len(hkl_refs)} theoretische Reflexe berechnet.")

            # Match peaks to hkl
            matched = match_peaks_to_hkl(peaks, hkl_refs, wavelength, tol=tol)
            matched_count = sum(1 for m in matched if m["h"] != "—")

            # Store in session state for export
            st.session_state["hkl_refs"] = hkl_refs
            st.session_state["matched"] = matched
            st.session_state["peaks"] = peaks
            st.session_state["tt_raw"] = tt_raw
            st.session_state["intens_raw"] = intens_raw

            # ─── 📈 Annotated Pattern ──────────────────────────
            st.markdown("### 📈 Annotiertes Diffraktogramm")
            fig_ann, ax_ann = plt.subplots(figsize=(12, 5))

            # Experimental full pattern
            if tt_raw is not None and intens_raw is not None:
                ax_ann.plot(tt_raw, intens_raw, color="#1f77b4", lw=1.0, label="Experiment")
                ax_ann.set_xlabel("2θ (°)", fontsize=11)
                ax_ann.set_ylabel("Intensität (a.u.)", fontsize=11)

            # Calculated sticks
            ref_tt = [r["2θ (°)"] for r in hkl_refs]
            ref_f = [r["|F|²"] for r in hkl_refs]
            if ref_f:
                ref_f_norm = np.array(ref_f) / max(ref_f) * 100
                ax_ann.vlines(ref_tt, 0, ref_f_norm, color="#1f77b4", lw=1.5,
                              alpha=0.6, label="Berechnet (|F|²)")

            # Matched peaks with hkl labels
            for m in matched:
                if m["h"] != "—":
                    tt_m = m["2θ obs"]
                    i_m = m["Intensität"]
                    label = f"{m['h']}{m['k']}{m['l']}"
                    # Scale marker size by intensity
                    ax_ann.scatter(tt_m, i_m, color="red", s=40, zorder=5)
                    ax_ann.annotate(label, (tt_m, i_m),
                                    textcoords="offset points", xytext=(0, 12),
                                    fontsize=7, ha="center", rotation=90,
                                    color="darkred", fontweight="bold")

            ax_ann.set_title("Diffraktogramm mit hkl-Zuordnung", fontsize=12)
            ax_ann.legend(fontsize=9)
            ax_ann.grid(True, alpha=0.3)
            fig_ann.tight_layout()
            st.pyplot(fig_ann)

            st.info(f"{matched_count}/{len(matched)} Peaks zugeordnet ({len(hkl_refs)} berechnete Reflexe)")

            # ─── 📋 Indexed Peaks & F(hkl) ────────────────────
            st.markdown("### 📋 Indizierte Peaks & F(hkl)")
            display_cols = ["2θ obs", "h", "k", "l", "d (Å)", "Δ2θ",
                            "|F|²", "|F|", "Intensität"]
            matched_df = [{k: m[k] for k in display_cols if k in m} for m in matched]
            st.dataframe(matched_df, use_container_width=True, hide_index=True)

            # ─── 🌀 Phase / Argand ─────────────────────────────
            st.markdown("### 🌀 Argand-Diagramm (komplexe Ebene)")
            fig_arg, ax_arg = plt.subplots(figsize=(6, 6))

            # Draw axes
            ax_arg.axhline(0, color="gray", lw=0.8)
            ax_arg.axvline(0, color="gray", lw=0.8)

            # Find matching hkl_refs
            matched_hkl_set = {(m["h"], m["k"], m["l"])
                               for m in matched if m["h"] != "—"}

            for r in hkl_refs:
                key = (r["h"], r["k"], r["l"])
                matched_flag = key in matched_hkl_set
                fr = r.get("F_real", 0)
                fi = r.get("F_imag", 0)
                color = "red" if matched_flag else "#1f77b4"
                size = 60 if matched_flag else 20
                alpha = 1.0 if matched_flag else 0.4
                ax_arg.scatter(fr, fi, c=color, s=size, alpha=alpha, zorder=5)
                if matched_flag:
                    ax_arg.annotate(f"{r['h']}{r['k']}{r['l']}",
                                    (fr, fi), fontsize=6, ha="center", va="bottom")

            # Circle guides
            max_r = max(math.sqrt(r.get("F_real", 0)**2 + r.get("F_imag", 0)**2)
                        for r in hkl_refs) * 1.1
            for r_val in [max_r * 0.25, max_r * 0.5, max_r * 0.75, max_r]:
                circle = plt.Circle((0, 0), r_val, fill=False, ls="--",
                                    lw=0.5, color="gray", alpha=0.3)
                ax_arg.add_patch(circle)

            ax_arg.set_xlim(-max_r, max_r)
            ax_arg.set_ylim(-max_r, max_r)
            ax_arg.set_aspect("equal")
            ax_arg.set_xlabel("Re(F) / F_real", fontsize=11)
            ax_arg.set_ylabel("Im(F) / F_imag", fontsize=11)
            ax_arg.set_title("Argand-Diagramm F(hkl)", fontsize=12)
            ax_arg.grid(True, alpha=0.3)
            fig_arg.tight_layout()
            st.pyplot(fig_arg)

            # ─── 📊 d-spacing Quality ──────────────────────────
            st.markdown("### 📊 d-spacing Abgleich")

            quality_data = []
            for m in matched:
                if m["h"] != "—":
                    # Find the calculated ref for this peak
                    for r in hkl_refs:
                        if r["h"] == m["h"] and r["k"] == m["k"] and r["l"] == m["l"]:
                            d_obs = m.get("d (Å)", 0)
                            d_calc = r["d (Å)"]
                            delta_d = abs(d_obs - d_calc)
                            delta_d_over_d = delta_d / d_calc * 100 if d_calc else 0
                            quality_data.append({
                                "hkl": f"{m['h']}{m['k']}{m['l']}",
                                "d_obs (Å)": d_obs,
                                "d_calc (Å)": d_calc,
                                "Δd (Å)": round(delta_d, 5),
                                "Δd/d (%)": round(delta_d_over_d, 3),
                                "2θ_obs": m["2θ obs"],
                                "2θ_calc": r["2θ (°)"],
                                "Δ2θ": m.get("Δ2θ", "—"),
                            })
                            break

            if quality_data:
                st.dataframe(quality_data, use_container_width=True, hide_index=True)

                # Statistics
                delta_d_vals = [q["Δd (Å)"] for q in quality_data]
                delta_d_over_d = [q["Δd/d (%)"] for q in quality_data]
                mean_dd = np.mean(delta_d_vals)
                max_dd = max(delta_d_vals)
                rms_dd = math.sqrt(np.mean(np.array(delta_d_vals)**2))

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Gelistete Reflexe", len(quality_data))
                col2.metric("Mittl. Δd", f"{mean_dd:.5f} Å")
                col3.metric("RMS Δd", f"{rms_dd:.5f} Å")
                col4.metric("Max Δd", f"{max_dd:.5f} Å")

                # Histogram of Δd/d
                fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
                ax_hist.hist(delta_d_over_d, bins=10, color="#1f77b4",
                             edgecolor="white", alpha=0.7)
                ax_hist.axvline(np.mean(delta_d_over_d), color="red", ls="--",
                                lw=1.5, label=f"Mittel: {np.mean(delta_d_over_d):.3f}%")
                ax_hist.set_xlabel("Δd/d (%)", fontsize=11)
                ax_hist.set_ylabel("Anzahl Reflexe", fontsize=11)
                ax_hist.set_title("d-spacing Abweichung", fontsize=12)
                ax_hist.legend(fontsize=9)
                ax_hist.grid(True, alpha=0.3)
                fig_hist.tight_layout()
                st.pyplot(fig_hist)
            else:
                st.info("Keine zugeordneten Reflexe für Qualitätsanalyse.")

            # ─── 🔬 Full HKL Table ─────────────────────────────
            st.markdown("### 🔬 Vollständige HKL-Tabelle")
            full_cols = ["h", "k", "l", "d (Å)", "2θ (°)",
                         "|F|²", "|F|", "F_real", "F_imag", "φ (°)"]
            full_df = [{k: r[k] for k in full_cols} for r in hkl_refs]
            st.dataframe(full_df, use_container_width=True, hide_index=True)

            # ─── 💾 Export ─────────────────────────────────────
            st.markdown("### 💾 Export")
            import io, csv, base64

            col_csv, col_txt = st.columns(2)
            with col_csv:
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=full_cols)
                writer.writeheader()
                writer.writerows(full_df)
                csv_str = buf.getvalue()
                b64 = base64.b64encode(csv_str.encode()).decode()
                href = f'<a href="data:text/csv;base64,{b64}" download="hkl_reflections.csv">📥 CSV herunterladen (alle Reflexe)</a>'
                st.markdown(href, unsafe_allow_html=True)

            with col_txt:
                match_cols = ["2θ obs", "h", "k", "l", "d (Å)", "Δ2θ", "|F|²", "Intensität"]
                match_df_export = [{k: m.get(k, "—") for k in match_cols} for m in matched]
                buf2 = io.StringIO()
                w2 = csv.DictWriter(buf2, fieldnames=match_cols)
                w2.writeheader()
                w2.writerows(match_df_export)
                txt_str = buf2.getvalue()
                b64_2 = base64.b64encode(txt_str.encode()).decode()
                href2 = f'<a href="data:text/csv;base64,{b64_2}" download="indexed_peaks.csv">📥 CSV herunterladen (indizierte Peaks)</a>'
                st.markdown(href2, unsafe_allow_html=True)

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

# ==================== TAB 5: FWHM + Scherrer ====================
with tab5:
    st.markdown("### 📐 FWHM-Diffraktogramm-Analyzer")
    st.markdown("Peak-Fitting, FWHM-Extraktion, Scherrer-Kristallitgröße")

    fwhm_file = st.file_uploader(
        "XRD-Datei (.xy, .txt, .csv)", type=["xy", "txt", "csv"], key="fwhm"
    )
    fwhm_wl = st.number_input(
        "λ (Å)", value=1.5406, format="%.4f",
        help="Cu Kα = 1.5406, Mo Kα = 0.7107"
    )
    fwhm_prominence = st.slider("Peak-Empfindlichkeit", 0.0, 0.5, 0.05, 0.01)
    scherrer_K = st.number_input("Scherrer-K (Formfaktor)", value=0.9, format="%.2f")

    if fwhm_file is not None:
        data = parse_xy(fwhm_file.read().decode("utf-8"))
        if data:
            tt, intens = data
            peaks = find_peaks(tt, intens, prominence=fwhm_prominence)

            if len(peaks) < 2:
                st.warning("Weniger als 2 Peaks gefunden — evtl. Empfindlichkeit erhöhen.")
            else:
                from scipy.optimize import curve_fit

                def gauss(x, A, mu, sigma, bg):
                    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + bg

                fwhm_results = []
                fit_curves = []
                tt_arr = np.array(tt)
                intens_arr = np.array(intens)

                # Sort peaks by 2θ
                peaks_sorted = sorted(peaks, key=lambda p: p[0])

                for idx, (p_tt, p_int) in enumerate(peaks_sorted):
                    # Window: ±2° around peak or to next/prev peak
                    half_window = 2.0
                    if idx > 0:
                        half_window = min(half_window, (p_tt - peaks_sorted[idx-1][0]) * 0.6)
                    if idx < len(peaks_sorted) - 1:
                        half_window = min(half_window, (peaks_sorted[idx+1][0] - p_tt) * 0.6)

                    mask = (tt_arr >= p_tt - half_window) & (tt_arr <= p_tt + half_window)
                    x_data = tt_arr[mask]
                    y_data = intens_arr[mask]

                    if len(x_data) < 5:
                        continue

                    # Initial guess
                    bg0 = np.min(y_data)
                    A0 = np.max(y_data) - bg0
                    sigma0 = 0.1
                    try:
                        popt, _ = curve_fit(
                            gauss, x_data, y_data,
                            p0=[A0, p_tt, sigma0, bg0],
                            maxfev=2000,
                        )
                        A_fit, mu_fit, sigma_fit, bg_fit = popt

                        if sigma_fit <= 0 or A_fit <= 0:
                            continue

                        fwhm = 2 * math.sqrt(2 * math.log(2)) * sigma_fit
                        fwhm_rad = math.radians(fwhm)

                        # Scherrer
                        theta_rad = math.radians(mu_fit / 2)
                        D = scherrer_K * fwhm_wl / (fwhm_rad * math.cos(theta_rad)) if fwhm_rad > 0 else 0

                        # R²
                        residuals = y_data - gauss(x_data, *popt)
                        ss_res = np.sum(residuals**2)
                        ss_tot = np.sum((y_data - np.mean(y_data))**2)
                        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                        fwhm_results.append({
                            "Peak": idx + 1,
                            "2θ (°)": round(mu_fit, 4),
                            "Intensität": round(A_fit + bg_fit, 1),
                            "FWHM (°)": round(fwhm, 4),
                            "FWHM (rad)": round(fwhm_rad, 6),
                            "σ": round(sigma_fit, 4),
                            "R²": round(r_sq, 4),
                            "D (nm)": round(D, 2) if D > 0 else 0,
                        })
                        fit_curves.append((x_data, gauss(x_data, *popt), mu_fit))

                    except (RuntimeError, ValueError):
                        continue

                if not fwhm_results:
                    st.error("Keine Peaks konnten gefittet werden.")
                else:
                    st.success(f"**{len(fwhm_results)} Peaks gefittet**")

                    # Stats row
                    fwhm_vals = [r["FWHM (°)"] for r in fwhm_results]
                    d_vals = [r["D (nm)"] for r in fwhm_results if r["D (nm)"] > 0]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Peaks gefunden & gefittet", len(fwhm_results))
                    c2.metric("Mittl. FWHM (°)", f"{np.mean(fwhm_vals):.4f}")
                    c3.metric("Min FWHM (°)", f"{min(fwhm_vals):.4f}")
                    c4.metric("Max FWHM (°)", f"{max(fwhm_vals):.4f}")

                    # ─── 📈 Diffractogram + Fits ────────────────
                    st.markdown("### 📈 Diffraktogramm + Fits")
                    fig_fit, ax_fit = plt.subplots(figsize=(12, 5))
                    ax_fit.plot(tt, intens, color="#1f77b4", lw=0.8,
                                label="Experiment", alpha=0.7)
                    for xf, yf, mu in fit_curves:
                        ax_fit.plot(xf, yf, color="red", lw=1.5)
                        ax_fit.axvline(mu, color="red", ls="--", lw=0.5, alpha=0.4)
                    ax_fit.set_xlabel("2θ (°)", fontsize=11)
                    ax_fit.set_ylabel("Intensität", fontsize=11)
                    ax_fit.set_title("Diffraktogramm mit Gauß-Fits", fontsize=12)
                    ax_fit.legend(["Experiment", "Gauß-Fit", "Peak-Zentrum"], fontsize=8)
                    ax_fit.grid(True, alpha=0.3)
                    fig_fit.tight_layout()
                    st.pyplot(fig_fit)

                    # ─── 📊 FWHM Results ────────────────────────
                    st.markdown("### 📊 FWHM-Ergebnisse")
                    fwhm_cols = ["Peak", "2θ (°)", "Intensität", "FWHM (°)",
                                 "FWHM (rad)", "σ", "R²", "D (nm)"]
                    fwhm_df = [{k: r[k] for k in fwhm_cols} for r in fwhm_results]
                    st.dataframe(fwhm_df, use_container_width=True, hide_index=True)

                    # ─── 📉 Scherrer & Williamson-Hall ───────────
                    st.markdown("### 📉 Scherrer & Williamson-Hall")

                    wh_data = [{
                        "Peak": r["Peak"],
                        "2θ (°)": r["2θ (°)"],
                        "FWHM (°)": r["FWHM (°)"],
                        "FWHM (rad)": r["FWHM (rad)"],
                        "D (nm)": r["D (nm)"],
                    } for r in fwhm_results if r["D (nm)"] > 0]

                    if wh_data:
                        st.dataframe(
                            [{"Peak": d["Peak"], "2θ (°)": d["2θ (°)"],
                              "FWHM (°)": d["FWHM (°)"], "D (nm)": d["D (nm)"]}
                             for d in wh_data],
                            use_container_width=True, hide_index=True
                        )

                        # Prepare WH variables
                        theta_arr = np.array([math.radians(d["2θ (°)"] / 2) for d in wh_data])
                        beta_arr = np.array([d["FWHM (rad)"] for d in wh_data])
                        x_wh = 4 * np.sin(theta_arr)            # 4 sinθ
                        y_wh = beta_arr * np.cos(theta_arr)     # β cosθ

                        # Linear fit: β cosθ = Kλ/D + 4ε sinθ
                        coeffs = np.polyfit(x_wh, y_wh, 1)
                        slope, intercept = coeffs
                        d_wh = scherrer_K * fwhm_wl / intercept if intercept > 0 else 0
                        strain = slope

                        x_fit = np.linspace(min(x_wh) * 0.8, max(x_wh) * 1.2, 50)
                        y_fit = np.polyval(coeffs, x_fit)

                        # ── Row: D vs 2θ + FWHM vs 2θ side by side ──
                        col_d, col_f = st.columns(2)

                        with col_d:
                            fig_d, ax_d = plt.subplots(figsize=(5, 4))
                            ax_d.scatter([d["2θ (°)"] for d in wh_data],
                                         [d["D (nm)"] for d in wh_data],
                                         color="red", s=60, zorder=5)
                            mean_d_val = np.mean([d["D (nm)"] for d in wh_data])
                            ax_d.axhline(mean_d_val, color="gray", ls="--", lw=1,
                                         label=f"Mittel: {mean_d_val:.1f} nm")
                            ax_d.set_xlabel("2θ (°)", fontsize=10)
                            ax_d.set_ylabel("D (nm)", fontsize=10)
                            ax_d.set_title("Kristallitgröße D vs. 2θ", fontsize=11)
                            ax_d.legend(fontsize=8)
                            ax_d.grid(True, alpha=0.3)
                            fig_d.tight_layout()
                            st.pyplot(fig_d)

                        with col_f:
                            fig_f, ax_f = plt.subplots(figsize=(5, 4))
                            ax_f.scatter([d["2θ (°)"] for d in wh_data],
                                         [d["FWHM (°)"] for d in wh_data],
                                         color="darkorange", s=60, zorder=5)
                            ax_f.set_xlabel("2θ (°)", fontsize=10)
                            ax_f.set_ylabel("FWHM (°)", fontsize=10)
                            ax_f.set_title("FWHM vs. 2θ", fontsize=11)
                            ax_f.grid(True, alpha=0.3)
                            fig_f.tight_layout()
                            st.pyplot(fig_f)

                        # ── Williamson-Hall plot ──
                        st.markdown("##### 🧱 Williamson-Hall: β·cosθ vs 4·sinθ")

                        fig_wh, ax_wh = plt.subplots(figsize=(7, 5))
                        ax_wh.scatter(x_wh, y_wh, color="darkgreen", s=70,
                                      zorder=5, label="Daten")
                        ax_wh.plot(x_fit, y_fit, color="red", lw=1.5,
                                   label=f"Fit: y={slope:.4f}x+{intercept:.4f}")
                        ax_wh.set_xlabel("4 sinθ", fontsize=11)
                        ax_wh.set_ylabel("β cosθ (rad)", fontsize=11)
                        ax_wh.set_title("Williamson-Hall Plot", fontsize=12)
                        ax_wh.legend(fontsize=9)
                        ax_wh.grid(True, alpha=0.3)
                        fig_wh.tight_layout()
                        st.pyplot(fig_wh)

                        # Metrics
                        c1_w, c2_w, c3_w, c4_w = st.columns(4)
                        c1_w.metric("D (Scherrer, Einzel)", f"{mean_d_val:.1f} nm")
                        c2_w.metric("D (W-H, aus Intercept)",
                                    f"{d_wh:.1f} nm" if d_wh > 0 else "—")
                        c3_w.metric("Mikrodehnung ε", f"{strain:.6f}")
                        c4_w.metric("R² (W-H Fit)",
                                    f"{np.corrcoef(x_wh, y_wh)[0,1]**2:.4f}")

                        # Instrumental broadening hint
                        st.caption(
                            "β = gemessene FWHM (rad). Für korrigierte Werte "
                            "β_korr² = β_gem² − β_inst² verwenden. "
                            "β_inst aus Standard (LaB₆, Si) bestimmen."
                        )
                    else:
                        st.info("Keine Kristallitgrößen berechnet (alle D=0?).")

                    # ─── 🔬 Individual Peaks ────────────────────
                    st.markdown("### 🔬 Einzelne Peaks")
                    n_peaks = len(fwhm_results)
                    # Show in a grid: 2 peaks per row
                    cols_per_row = 2
                    for row_start in range(0, n_peaks, cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            pi = row_start + j
                            if pi >= n_peaks:
                                break
                            r = fwhm_results[pi]
                            with cols[j]:
                                # Extract the fit curve for this peak
                                _, xf_plot, _ = fit_curves[pi]

                                # Find corresponding data window
                                mu = r["2θ (°)"]
                                mask = (tt_arr >= mu - 2) & (tt_arr <= mu + 2)
                                xw = tt_arr[mask]
                                yw = intens_arr[mask]

                                if len(xw) < 3:
                                    continue

                                fig_pi, ax_pi = plt.subplots(figsize=(5, 3))
                                ax_pi.plot(xw, yw, color="#1f77b4", lw=1, label="Data")
                                # Re-fit just for this window for the plot
                                mask2 = (tt_arr >= mu - 1.5) & (tt_arr <= mu + 1.5)
                                xw2 = tt_arr[mask2]
                                yw2 = intens_arr[mask2]
                                if len(xw2) >= 5:
                                    try:
                                        popt2, _ = curve_fit(
                                            gauss, xw2, yw2,
                                            p0=[r["Intensität"], mu, 0.1, np.min(yw2)],
                                            maxfev=2000
                                        )
                                        ax_pi.plot(xw2, gauss(xw2, *popt2), "red",
                                                   lw=1.5, label="Fit")
                                    except Exception:
                                        pass
                                ax_pi.axvline(mu, color="red", ls="--", lw=0.8, alpha=0.5)
                                ax_pi.set_title(f"Peak {r['Peak']}: {mu:.2f}°\n"
                                                f"FWHM={r['FWHM (°)']:.4f}°  "
                                                f"D={r['D (nm)']:.1f}nm",
                                                fontsize=9)
                                ax_pi.set_xlabel("2θ (°)", fontsize=8)
                                ax_pi.set_ylabel("Intensität", fontsize=8)
                                ax_pi.tick_params(labelsize=7)
                                ax_pi.grid(True, alpha=0.3)
                                fig_pi.tight_layout()
                                st.pyplot(fig_pi)

                    # ─── 💾 Export ─────────────────────────────
                    st.markdown("### 💾 Export")
                    import io, csv, base64
                    buf = io.StringIO()
                    w = csv.DictWriter(buf, fieldnames=fwhm_cols)
                    w.writeheader()
                    w.writerows(fwhm_df)
                    b64 = base64.b64encode(buf.getvalue().encode()).decode()
                    href = f'<a href="data:text/csv;base64,{b64}" download="fwhm_results.csv">📥 FWHM-Ergebnisse als CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)

        else:
            st.error("Datei konnte nicht geparst werden.")

st.caption("FellX4 — mit HKL-Suche & Strukturfaktoren 🚀🔷")
