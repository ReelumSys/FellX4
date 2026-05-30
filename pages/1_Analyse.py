     1|     1|import streamlit as st
     2|     2|import matplotlib.pyplot as plt
     3|     3|import numpy as np
     4|     4|import re
     5|     5|import math
     6|     6|import io
     7|     7|import csv
     8|     8|import base64
     9|     9|
    10|    10|st.set_page_config(page_title="FellX4 — Analyse", layout="wide")
    11|    11|
    12|    12|# ──────────────────────────────────────────────
    13|    13|#  Prüfen: Daten da?
    14|    14|# ──────────────────────────────────────────────
    15|    15|if "crystal" not in st.session_state or "tt_raw" not in st.session_state:
    16|    16|    st.warning("⚠️ Keine Daten geladen — bitte zuerst CIF + Diffraktogramm auf der Startseite hochladen.")
    17|    17|    if st.button("← Neue Daten laden"):
    18|    18|        st.switch_page("pages/home.py")
    19|    19|    st.stop()
    20|    20|
    21|    21|crystal = st.session_state["crystal"]
    22|    22|a = st.session_state["a"]
    23|    23|b = st.session_state["b"]
    24|    24|c = st.session_state["c"]
    25|    25|alpha = st.session_state["alpha"]
    26|    26|beta = st.session_state["beta"]
    27|    27|gamma = st.session_state["gamma"]
    28|    28|sg = st.session_state["sg"]
    29|    29|wavelength = st.session_state.get("wavelength", 1.5406)
    30|    30|tt_raw = st.session_state["tt_raw"]
    31|    31|intens_raw = st.session_state["intens_raw"]
    32|    32|peaks = st.session_state.get("peaks", [])
    33|    33|cif_name = st.session_state.get("cif_file_name", "?")
    34|    34|xrd_name = st.session_state.get("xrd_file_name", "?")
    35|    35|
    36|    36|# ──────────────────────────────────────────────
    37|    37|#  Hilfsfunktionen
    38|    38|# ──────────────────────────────────────────────
    39|    39|
    40|    40|def parse_xy(content: str):
    41|    41|    data = []
    42|    42|    for line in content.strip().splitlines():
    43|    43|        line = line.strip()
    44|    44|        if not line or line.startswith("#") or line.startswith(";"):
    45|    45|            continue
    46|    46|        parts = line.split()
    47|    47|        if len(parts) >= 2:
    48|    48|            try:
    49|    49|                data.append((float(parts[0]), float(parts[1])))
    50|    50|            except ValueError:
    51|    51|                continue
    52|    52|    if not data:
    53|    53|        return None
    54|    54|    tt, intens = zip(*data)
    55|    55|    return list(tt), list(intens)
    56|    56|
    57|    57|def plot_xy(tt, intens, title="Diffraktogramm", color="#1f77b4", ax=None):
    58|    58|    if ax is None:
    59|    59|        fig, ax = plt.subplots(figsize=(10, 4))
    60|    60|    ax.plot(tt, intens, color=color, linewidth=1.0)
    61|    61|    ax.set_xlabel("2θ (°)", fontsize=11)
    62|    62|    ax.set_ylabel("Intensität (a.u.)", fontsize=11)
    63|    63|    ax.set_title(title, fontsize=12)
    64|    64|    ax.grid(True, alpha=0.3)
    65|    65|    fig = ax.figure
    66|    66|    fig.tight_layout()
    67|    67|    return fig
    68|    68|
    69|    69|def find_peaks(tt, intens, prominence=0.05):
    70|    70|    from scipy.signal import find_peaks as sp_find_peaks
    71|    71|    from scipy.ndimage import gaussian_filter1d
    72|    72|    arr = np.array(intens, dtype=float)
    73|    73|    if np.max(arr) == 0:
    74|    74|        return []
    75|    75|    smoothed = gaussian_filter1d(arr, sigma=1.5)
    76|    76|    abs_prominence = prominence * np.max(arr)
    77|    77|    if len(tt) > 1:
    78|    78|        step = np.median(np.diff(tt))
    79|    79|        distance = max(3, int(0.5 / step))
    80|    80|    else:
    81|    81|        distance = 3
    82|    82|    peaks_idx, _ = sp_find_peaks(smoothed, prominence=abs_prominence, distance=distance, width=1)
    83|    83|    return [(tt[i], intens[i]) for i in peaks_idx]
    84|    84|
    85|    85|def d_spacing(h, k, l, a, b, c, alpha, beta, gamma):
    86|    86|    al = math.radians(alpha)
    87|    87|    be = math.radians(beta)
    88|    88|    ga = math.radians(gamma)
    89|    89|    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    90|    90|    sa, sb, sg = math.sin(al), math.sin(be), math.sin(ga)
    91|    91|    vol = a * b * c * math.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
    92|    92|    a_star = b * c * sa / vol
    93|    93|    b_star = a * c * sb / vol
    94|    94|    c_star = a * b * sg / vol
    95|    95|    ca_star = (cb * cg - ca) / (sb * sg)
    96|    96|    cb_star = (ca * cg - cb) / (sa * sg)
    97|    97|    cg_star = (ca * cb - cg) / (sa * sb)
    98|    98|    d2 = (h**2 * a_star**2 + k**2 * b_star**2 + l**2 * c_star**2
    99|    99|          + 2 * h * k * a_star * b_star * cg_star
   100|   100|          + 2 * h * l * a_star * c_star * cb_star
   101|   101|          + 2 * k * l * b_star * c_star * ca_star)
   102|   102|    if d2 <= 0:
   103|   103|        return None
   104|   104|    return 1.0 / math.sqrt(d2)
   105|   105|
   106|   106|SCATTERING_FACTORS = {
   107|   107|    "H":  [(0.493, 10.511), (0.323, 26.126), (0.140, 3.142), (0.041, 57.800), 0.003],
   108|   108|    "C":  [(2.310, 20.844), (1.020, 10.208), (1.589, 0.569), (0.865, 51.651), 0.216],
   109|   109|    "N":  [(12.213, 0.006), (3.132, 9.893), (2.013, 28.665), (1.166, 0.396), -11.529],
   110|   110|    "O":  [(3.049, 13.277), (2.287, 5.701), (1.546, 0.324), (0.867, 32.909), 0.251],
   111|   111|    "Na": [(4.763, 3.285), (3.174, 8.842), (1.268, 0.314), (1.113, 129.424), 0.676],
   112|   112|    "Mg": [(5.420, 2.828), (2.174, 79.261), (1.227, 0.381), (0.859, 21.806), 0.318],
   113|   113|    "Al": [(6.420, 3.039), (1.594, 77.558), (1.465, 0.402), (1.043, 21.024), 0.477],
   114|   114|    "Si": [(6.292, 2.439), (3.035, 32.334), (1.989, 0.678), (1.541, 81.694), 0.145],
   115|   115|    "P":  [(6.435, 1.907), (4.179, 27.157), (1.781, 0.647), (1.165, 67.913), 0.442],
   116|   116|    "S":  [(6.905, 1.468), (5.203, 22.215), (1.438, 0.254), (1.586, 55.925), 0.867],
   117|   117|    "Cl": [(11.460, 0.010), (7.196, 1.166), (6.256, 18.520), (1.646, 47.778), -9.557],
   118|   118|    "K":  [(8.219, 12.795), (7.440, 0.775), (1.052, 213.719), (0.866, 41.684), 0.424],
   119|   119|    "Ca": [(8.627, 10.442), (7.387, 0.660), (1.590, 85.748), (1.021, 178.438), 0.375],
   120|   120|    "Ti": [(9.759, 7.851), (7.356, 0.472), (1.699, 37.267), (1.203, 111.638), 0.982],
   121|   121|    "Fe": [(11.776, 1.035), (7.122, 11.441), (4.148, 0.656), (2.400, 53.144), 0.557],
   122|   122|    "Ni": [(12.838, 1.503), (7.292, 11.395), (4.284, 0.462), (2.255, 50.719), 0.333],
   123|   123|    "Cu": [(13.338, 3.583), (7.168, 0.231), (5.616, 14.079), (2.263, 49.259), 0.616],
   124|   124|    "Zn": [(14.074, 3.265), (7.032, 0.233), (5.165, 12.895), (2.411, 44.571), 1.319],
   125|   125|    "Sr": [(19.215, 17.462), (16.360, 2.661), (4.077, 71.450), (2.361, 0.013), 0.981],
   126|   126|    "Ba": [(24.307, 0.002), (17.635, 16.115), (6.992, 0.430), (4.350, 70.900), 0.718],
   127|   127|    "Pb": [(32.366, 0.033), (22.675, 10.241), (9.404, 0.513), (5.462, 52.131), 4.095],
   128|   128|}
   129|   129|
   130|   130|def atomic_scattering_factor(element: str, sin_theta_over_lambda: float) -> float:
   131|   131|    coeffs = SCATTERING_FACTORS.get(element.capitalize())
   132|   132|    if coeffs is None:
   133|   133|        return 0.0
   134|   134|    s2 = sin_theta_over_lambda**2
   135|   135|    f = coeffs[4]
   136|   136|    for a_i, b_i in coeffs[:4]:
   137|   137|        f += a_i * math.exp(-b_i * s2)
   138|   138|    return f
   139|   139|
   140|   140|def compute_structure_factors(atoms, a, b, c, alpha, beta, gamma, wavelength,
   141|   141|                               hkl_ranges=(10, 10, 10), min_f_sq_frac=0.01):
   142|   142|    h_max, k_max, l_max = hkl_ranges
   143|   143|    raw_results = []
   144|   144|    for h in range(-h_max, h_max + 1):
   145|   145|        for k in range(-k_max, k_max + 1):
   146|   146|            for l in range(-l_max, l_max + 1):
   147|   147|                if h == 0 and k == 0 and l == 0:
   148|   148|                    continue
   149|   149|                d = d_spacing(h, k, l, a, b, c, alpha, beta, gamma)
   150|   150|                if d is None or d <= 0:
   151|   151|                    continue
   152|   152|                sintheta = wavelength / (2 * d)
   153|   153|                if abs(sintheta) > 1:
   154|   154|                    continue
   155|   155|                theta = math.degrees(math.asin(sintheta))
   156|   156|                two_theta = 2 * theta
   157|   157|                if two_theta < 5 or two_theta > 150:
   158|   158|                    continue
   159|   159|                sin_t_over_l = math.sin(math.radians(theta)) / wavelength
   160|   160|                F_real = 0.0; F_imag = 0.0
   161|   161|                for atom in atoms:
   162|   162|                    try:
   163|   163|                        x = float(atom.get("fract_x", 0))
   164|   164|                        y = float(atom.get("fract_y", 0))
   165|   165|                        z = float(atom.get("fract_z", 0))
   166|   166|                        occ = float(atom.get("occupancy", 1.0))
   167|   167|                        match = re.match(r"([A-Za-z]+)", atom.get("type_symbol", "H"))
   168|   168|                        element = match.group(1) if match else "H"
   169|   169|                    except (ValueError, AttributeError):
   170|   170|                        continue
   171|   171|                    f_j = atomic_scattering_factor(element, sin_t_over_l)
   172|   172|                    phase = 2 * math.pi * (h * x + k * y + l * z)
   173|   173|                    F_real += occ * f_j * math.cos(phase)
   174|   174|                    F_imag += occ * f_j * math.sin(phase)
   175|   175|                F_sq = F_real**2 + F_imag**2
   176|   176|                if F_sq > 0.001:
   177|   177|                    phase_rad = math.atan2(F_imag, F_real)
   178|   178|                    raw_results.append({
   179|   179|                        "h": h, "k": k, "l": l, "d (Å)": round(d, 4),
   180|   180|                        "2θ (°)": round(two_theta, 4), "|F|²": round(F_sq, 2),
   181|   181|                        "|F|": round(math.sqrt(F_sq), 2), "F_real": round(F_real, 3),
   182|   182|                        "F_imag": round(F_imag, 3), "φ (°)": round(math.degrees(phase_rad), 1),
   183|   183|                    })
   184|   184|    if raw_results:
   185|   185|        max_f = max(r["|F|²"] for r in raw_results)
   186|   186|        threshold = min_f_sq_frac * max_f
   187|   187|        results = [r for r in raw_results if r["|F|²"] >= threshold]
   188|   188|    else:
   189|   189|        results = []
   190|   190|    results.sort(key=lambda r: r["d (Å)"], reverse=True)
   191|   191|    return results
   192|   192|
   193|   193|def match_peaks_to_hkl(peaks, hkl_data, wavelength, tol=0.2):
   194|   194|    matched = []
   195|   195|    for tt_obs, intens in peaks:
   196|   196|        best = None; best_delta = float("inf")
   197|   197|        for ref in hkl_data:
   198|   198|            delta = abs(ref["2θ (°)"] - tt_obs)
   199|   199|            if delta < best_delta and delta < tol:
   200|   200|                best_delta = delta; best = ref
   201|   201|        matched.append({"2θ obs": round(tt_obs, 4), "Intensität": round(intens, 2),
   202|   202|                        "h": best["h"] if best else "—", "k": best["k"] if best else "—",
   203|   203|                        "l": best["l"] if best else "—", "d (Å)": best["d (Å)"] if best else "—",
   204|   204|                        "Δ2θ": round(best_delta, 4) if best else "—",
   205|   205|                        "|F|": best["|F|"] if best else "—"})
   206|   206|    return matched
   207|   207|
   208|   208|def frac_to_cart(frac, a, b, c, alpha, beta, gamma):
   209|   209|    ca = math.cos(math.radians(alpha)); cb = math.cos(math.radians(beta))
   210|   210|    cg = math.cos(math.radians(gamma)); sg = math.sin(math.radians(gamma))
   211|   211|    ax_v = a; bx = b * cg; by = b * sg; cx = c * cb
   212|   212|    cy = c * (ca - cb * cg) / sg
   213|   213|    cz = c * math.sqrt(1 - cb**2 - ((ca - cb * cg) / sg)**2)
   214|   214|    x, y, z = frac
   215|   215|    return (x * ax_v + y * bx + z * cx, y * by + z * cy, z * cz)
   216|   216|
   217|   217|def plot_unit_cell(atoms, a, b, c, alpha, beta, gamma):
   218|   218|    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
   219|   219|    corners_frac = [(0,0,0), (1,0,0), (1,1,0), (0,1,0),
   220|   220|                    (0,0,1), (1,0,1), (1,1,1), (0,1,1)]
   221|   221|    corners = [frac_to_cart(cr, a, b, c, alpha, beta, gamma) for cr in corners_frac]
   222|   222|    fig = plt.figure(figsize=(7, 6))
   223|   223|    ax = fig.add_subplot(111, projection="3d")
   224|   224|    idx = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
   225|   225|    for i, j in idx:
   226|   226|        ax.plot([corners[i][0], corners[j][0]], [corners[i][1], corners[j][1]],
   227|   227|                [corners[i][2], corners[j][2]], color="#1f77b4", lw=1.5)
   228|   228|    faces_idx = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)]
   229|   229|    face_verts = [[corners[i] for i in f] for f in faces_idx]
   230|   230|    poly = Poly3DCollection(face_verts, alpha=0.05, facecolor="#1f77b4", edgecolor="none")
   231|   231|    ax.add_collection3d(poly)
   232|   232|    colors = {"H":"#ffffff","C":"#222222","N":"#3050f8","O":"#ff0d0d","Na":"#ab5cf2",
   233|   233|              "Mg":"#8aff00","Al":"#bfa6a6","Si":"#f0c8a0","P":"#ff8000","S":"#ffff30",
   234|   234|              "Cl":"#1ff01f","K":"#8f40d4","Ca":"#3dff00","Ti":"#bfc2c7","Fe":"#e06633",
   235|   235|              "Ni":"#50f050","Cu":"#c88033","Zn":"#71d0a0","Sr":"#00ff00","Ba":"#00c900","Pb":"#575961"}
   236|   236|    for atom in atoms:
   237|   237|        try:
   238|   238|            fx = float(atom.get("fract_x", 0)); fy = float(atom.get("fract_y", 0))
   239|   239|            fz = float(atom.get("fract_z", 0)); occ = float(atom.get("occupancy", 1.0))
   240|   240|            match = re.match(r"([A-Za-z]+)", atom.get("type_symbol", "H"))
   241|   241|            el = match.group(1) if match else "H"
   242|   242|        except: continue
   243|   243|        pos = frac_to_cart((fx, fy, fz), a, b, c, alpha, beta, gamma)
   244|   244|        ax.scatter(*pos, c=colors.get(el.capitalize(), "#888888"),
   245|   245|                   s=80+40*(occ if occ<=1 else 1), edgecolors="black", linewidths=0.3, alpha=0.85, zorder=10)
   246|   246|    av = frac_to_cart((1.15,0,0), a, b, c, alpha, beta, gamma)
   247|   247|    bv = frac_to_cart((0,1.15,0), a, b, c, alpha, beta, gamma)
   248|   248|    cv = frac_to_cart((0,0,1.15), a, b, c, alpha, beta, gamma)
   249|   249|    ax.text(*av, "a", fontsize=12, fontweight="bold", color="#1f77b4")
   250|   250|    ax.text(*bv, "b", fontsize=12, fontweight="bold", color="#1f77b4")
   251|   251|    ax.text(*cv, "c", fontsize=12, fontweight="bold", color="#1f77b4")
   252|   252|    mx = np.max(np.abs(np.array(corners)))
   253|   253|    ax.set_xlim(-mx*0.1, mx*1.1); ax.set_ylim(-mx*0.1, mx*1.1); ax.set_zlim(-mx*0.1, mx*1.1)
   254|   254|    ax.set_box_aspect([1,1,1]); ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
   255|   255|    ax.set_title("Elementarzelle", fontsize=12, fontweight="bold"); fig.tight_layout()
   256|   256|    return fig
   257|   257|
   258|   258|# ─── Rietveld helpers ───
   259|   259|from scipy.optimize import least_squares
   260|   260|
   261|   261|def pseudo_voigt(x, mu, fwhm, eta=0.5):
   262|   262|    if fwhm <= 0: return 0.0
   263|   263|    s = fwhm / (2 * np.sqrt(2 * np.log(2)))
   264|   264|    # Gaussian
   265|   265|    g = np.exp(-0.5 * ((x - mu) / s)**2) / (s * np.sqrt(2 * np.pi))
   266|   266|    # Lorentzian
   267|   267|    l = fwhm / (2 * np.pi) / ((x - mu)**2 + (fwhm / 2)**2)
   268|   268|    return eta * l + (1 - eta) * g
   269|   269|
   270|   270|def caglioti_fwhm(theta_deg, U, V, W):
   271|   271|    t = np.tan(np.radians(theta_deg))
   272|   272|    return np.sqrt(np.maximum(U * t**2 + V * t + W, 0.001))
   273|   273|
   274|   274|def riet_calc_pattern(params_dict, tt_obs, hkl_refs, bg_order, wavelength, alpha, beta, gamma):
   275|   275|    scale = params_dict['scale']
   276|   276|    zshift = params_dict['zshift']
   277|   277|    U, V, W = params_dict['U'], params_dict['V'], params_dict['W']
   278|   278|    a, b, c = params_dict['a'], params_dict['b'], params_dict['c']
   279|   279|    bg_coeffs = params_dict['bg']
   280|   280|
   281|   281|    y_calc = np.zeros_like(tt_obs, dtype=float)
   282|   282|    for r in hkl_refs:
   283|   283|        d = d_spacing(r["h"], r["k"], r["l"], a, b, c, alpha, beta, gamma)
   284|   284|        if d is None or d <= 0: continue
   285|   285|        stheta = wavelength / (2 * d)
   286|   286|        if abs(stheta) > 1: continue
   287|   287|        tt_pos = (2 * np.degrees(np.arcsin(stheta))) + zshift
   288|   288|        
   289|   289|        # Peak-Profil über den gesamten tt_obs Bereich
   290|   290|        fwhm_val = caglioti_fwhm(tt_pos, U, V, W)
   291|   291|        y_calc += r["|F|²"] * scale * pseudo_voigt(tt_obs, tt_pos, fwhm_val)
   292|   292|
   293|   293|    # Background Polynomial
   294|   294|    x_norm = (tt_obs - np.min(tt_obs)) / (np.max(tt_obs) - np.min(tt_obs) + 1e-10)
   295|   295|    bg = np.zeros_like(tt_obs)
   296|   296|    for i, coeff in enumerate(bg_coeffs):
   297|   297|        bg += coeff * (x_norm**i)
   298|   298|    
   299|   299|    return y_calc + np.maximum(bg, 0)
   300|   300|
   301|   301|def riet_residuals(x_var, var_keys, current_params, tt_obs, y_obs, hkl_refs, bg_order, wavelength, alpha, beta, gamma):
   302|   302|    # Map flat vector x_var back to the dictionary current_params
   303|   303|    params = current_params.copy()
   304|   304|    for i, key in enumerate(var_keys):
   305|   305|        if key.startswith('bg'): 
   306|   306|            # Background keys are handled as bg_coeffs list
   307|   307|            # We'll handle this by updating the 'bg' list in params
   308|   308|            bg_idx = int(key[2:])
   309|   309|            if 'bg' not in params or not isinstance(params['bg'], list):
   310|   310|                params['bg'] = [0.0] * (bg_order + 1)
   311|   311|            params['bg'][bg_idx] = x_var[i]
   312|   312|        else:
   313|   313|            params[key] = x_var[i]
   314|   314|    
   315|   315|    y_calc = riet_calc_pattern(params, tt_obs, hkl_refs, bg_order, wavelength, alpha, beta, gamma)
   316|   316|    w = 1.0 / np.sqrt(np.maximum(y_obs, 1.0))
   317|   317|    return (y_obs - y_calc) * w
   318|   318|
   319|   319|# ──────────────────────────────────────────────
   320|   320|#  Top Info + Navigation
   321|   321|# ──────────────────────────────────────────────
   322|   322|st.title("🔬 FellX4 — Analyse")
   323|   323|col_info, col_back = st.columns([3, 1])
   324|   324|with col_info:
   325|   325|    st.markdown(f"**CIF:** {cif_name}  ·  **XRD:** {xrd_name}  ·  "
   326|   326|                f"a={a:.3f} b={b:.3f} c={c:.3f}  ·  "
   327|   327|                f"λ={wavelength:.4f} Å  ·  {len(peaks)} Peaks")
   328|   328|with col_back:
   329|   329|    if st.button("← Neue Daten laden"):
   330|   330|        st.switch_page("pages/home.py")
   331|   331|
   332|   332|# ──────────────────────────────────────────────
   333|   333|#  TABS
   334|   334|# ──────────────────────────────────────────────
   335|   335|tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
   336|   336|    "📁 Diffraktogramm",
   337|   337|    "📊 Vergleich + Subtraktion",
   338|   338|    "🔷 HKL + Strukturfaktoren",
   339|   339|    "✏️ Manuelle Eingabe",
   340|   340|    "📐 FWHM + Scherrer",
   341|   341|    "🧪 Rietveld",
   342|   342|])
   343|   343|
   344|   344|# ==================== TAB 1 ====================
   345|   345|with tab1:
   346|   346|    st.pyplot(plot_xy(tt_raw, intens_raw, title=xrd_name))
   347|   347|    with st.expander("📄 Rohdaten"):
   348|   348|        st.dataframe({"2θ (°)": tt_raw, "Intensität": intens_raw}, use_container_width=True)
   349|   349|
   350|   350|# ==================== TAB 2 ====================
   351|   351|with tab2:
   352|   352|    col_a, col_b = st.columns(2)
   353|   353|    with col_a: fa = st.file_uploader("Datei A", type=["xy", "txt", "csv"], key="c2a")
   354|   354|    with col_b: fb = st.file_uploader("Datei B", type=["xy", "txt", "csv"], key="c2b")
   355|   355|    if fa and fb:
   356|   356|        da = parse_xy(fa.read().decode("utf-8")); db = parse_xy(fb.read().decode("utf-8"))
   357|   357|        if da and db:
   358|   358|            x1, y1 = da; x2, y2 = db
   359|   359|            if len(x1) >= len(x2):
   360|   360|                y2i = np.interp(x1, x2, y2); diff = np.array(y1) - y2i; xd = x1
   361|   361|            else:
   362|   362|                y1i = np.interp(x2, x1, y1); diff = np.array(y1i) - y2; xd = x2
   363|   363|            fig_cmp, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
   364|   364|                                                gridspec_kw={"height_ratios": [2, 1]})
   365|   365|            ax1.plot(x1, y1, color="#1f77b4", lw=1.0, label=f"A: {fa.name}")
   366|   366|            ax1.plot(x2, y2, color="#d62728", lw=1.0, label=f"B: {fb.name}")
   367|   367|            ax1.set_ylabel("Intensität"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
   368|   368|            ax2.axhline(0, color="gray", lw=0.8, ls="--")
   369|   369|            ax2.plot(xd, diff, color="#2ca02c", lw=1.0)
   370|   370|            ax2.set_xlabel("2θ (°)"); ax2.set_ylabel("A − B"); ax2.grid(True, alpha=0.3)
   371|   371|            fig_cmp.tight_layout(); st.pyplot(fig_cmp)
   372|   372|    elif not fa and not fb:
   373|   373|        st.info("Lade beide Dateien für den Vergleich.")
   374|   374|
   375|   375|# ==================== TAB 3 ====================
   376|   376|with tab3:
   377|   377|    st.markdown("### 📄 CIF-Struktur")
   378|   378|    st.success(f"✅ {cif_name} — a={a:.3f} b={b:.3f} c={c:.3f} Å, "
   379|   379|               f"α={alpha:.2f} β={beta:.2f} γ={gamma:.2f}°, {sg} | {len(crystal['atoms'])} Atome")
   380|   380|    with st.expander("📄 Atompositionen"):
   381|   381|        st.dataframe(crystal["atoms"], use_container_width=True)
   382|   382|    if st.checkbox("🔬 3D-Elementarzelle anzeigen", value=True):
   383|   383|        st.pyplot(plot_unit_cell(crystal["atoms"], a, b, c, alpha, beta, gamma))
   384|   384|
   385|   385|    st.markdown("---"); st.markdown("### 📊 Diffraktogramm (experimentell)")
   386|   386|    prominence = st.slider("Peak-Empfindlichkeit", 0.0, 0.5, 0.05, 0.01, key="hkl_prom")
   387|   387|    local_peaks = find_peaks(tt_raw, intens_raw, prominence=prominence)
   388|   388|    fig_p, (ax_p, ax_peaks) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
   389|   389|                                            gridspec_kw={"height_ratios": [3, 1]})
   390|   390|    ax_p.plot(tt_raw, intens_raw, color="#1f77b4", lw=1.0)
   391|   391|    pt = [p[0] for p in local_peaks]; pi = [p[1] for p in local_peaks]
   392|   392|    ax_p.scatter(pt, pi, color="red", s=30, zorder=5, label=f"{len(local_peaks)} Peaks")
   393|   393|    ax_p.set_ylabel("Intensität"); ax_p.legend(fontsize=9); ax_p.grid(True, alpha=0.3)
   394|   394|    ax_peaks.bar(pt, pi, width=0.08, color="red", alpha=0.6)
   395|   395|    ax_peaks.set_xlabel("2θ (°)"); ax_peaks.grid(True, alpha=0.3)
   396|   396|    fig_p.tight_layout(); st.pyplot(fig_p)
   397|   397|    st.info(f"{len(local_peaks)} Peaks detektiert")
   398|   398|
   399|   399|    hkl_range = st.slider("hkl-Suchbereich", 1, 10, 5, key="hkl_range")
   400|   400|    tol = st.slider("Toleranz Δ2θ (°)", 0.05, 1.0, 0.3, 0.05, key="hkl_tol")
   401|   401|    st.markdown("---")
   402|   402|
   403|   403|    if st.button("🧮 hkl-Indizierung + Strukturfaktoren berechnen", type="primary"):
   404|   404|        if not local_peaks:
   405|   405|            st.error("❌ Keine Peaks — Empfindlichkeit erhöhen.")
   406|   406|        else:
   407|   407|            with st.spinner("Berechne hkl-Reflexe..."):
   408|   408|                hkl_refs = compute_structure_factors(
   409|   409|                    crystal["atoms"], a, b, c, alpha, beta, gamma,
   410|   410|                    wavelength, (hkl_range, hkl_range, hkl_range))
   411|   411|            st.success(f"✅ {len(hkl_refs)} theoretische Reflexe berechnet.")
   412|   412|            matched = match_peaks_to_hkl(local_peaks, hkl_refs, wavelength, tol=tol)
   413|   413|            matched_count = sum(1 for m in matched if m["h"] != "—")
   414|   414|            st.session_state["hkl_refs"] = hkl_refs
   415|   415|            st.session_state["matched"] = matched
   416|   416|            st.session_state["local_peaks"] = local_peaks
   417|   417|
   418|   418|            st.markdown("### 📈 Annotiertes Diffraktogramm")
   419|   419|            fig_ann, ax_ann = plt.subplots(figsize=(12, 5))
   420|   420|            ax_ann.plot(tt_raw, intens_raw, color="#1f77b4", lw=1.0, label="Experiment")
   421|   421|            ref_tt = [r["2θ (°)"] for r in hkl_refs]
   422|   422|            ref_f = [r["|F|²"] for r in hkl_refs]
   423|   423|            if ref_f:
   424|   424|                ax_ann.vlines(ref_tt, 0, np.array(ref_f)/max(ref_f)*100, color="#1f77b4", lw=1.5, alpha=0.6, label="Berechnet")
   425|   425|            for m in matched:
   426|   426|                if m["h"] != "—":
   427|   427|                    ax_ann.scatter(m["2θ obs"], m["Intensität"], color="red", s=40, zorder=5)
   428|   428|                    ax_ann.annotate(f"{m['h']}{m['k']}{m['l']}", (m["2θ obs"], m["Intensität"]),
   429|   429|                                    textcoords="offset points", xytext=(0, 12), fontsize=7, ha="center", rotation=90,
   430|   430|                                    color="darkred", fontweight="bold")
   431|   431|            ax_ann.set_xlabel("2θ (°)"); ax_ann.set_ylabel("Intensität")
   432|   432|            ax_ann.set_title("Diffraktogramm mit hkl-Zuordnung", fontsize=12)
   433|   433|            ax_ann.legend(fontsize=9); ax_ann.grid(True, alpha=0.3); fig_ann.tight_layout(); st.pyplot(fig_ann)
   434|   434|            st.info(f"{matched_count}/{len(matched)} Peaks zugeordnet ({len(hkl_refs)} berechnete Reflexe)")
   435|   435|
   436|   436|            st.markdown("### 📋 Indizierte Peaks & F(hkl)")
   437|   437|            disp = ["2θ obs", "h", "k", "l", "d (Å)", "Δ2θ", "|F|²", "|F|", "Intensität"]
   438|   438|            st.dataframe([{k: m[k] for k in disp if k in m} for m in matched], use_container_width=True, hide_index=True)
   439|   439|
   440|   440|            st.markdown("### 🌀 Argand-Diagramm")
   441|   441|            fig_arg, ax_arg = plt.subplots(figsize=(6, 6))
   442|   442|            ax_arg.axhline(0, color="gray", lw=0.8); ax_arg.axvline(0, color="gray", lw=0.8)
   443|   443|            mh = {(m["h"], m["k"], m["l"]) for m in matched if m["h"] != "—"}
   444|   444|            for r in hkl_refs:
   445|   445|                key = (r["h"], r["k"], r["l"]); flag = key in mh
   446|   446|                ax_arg.scatter(r["F_real"], r["F_imag"], c="red" if flag else "#1f77b4",
   447|   447|                               s=60 if flag else 20, alpha=1.0 if flag else 0.4, zorder=5)
   448|   448|                if flag: ax_arg.annotate(f"{r['h']}{r['k']}{r['l']}", (r["F_real"], r["F_imag"]), fontsize=6, ha="center", va="bottom")
   449|   449|            max_r = max(math.sqrt(r["F_real"]**2 + r["F_imag"]**2) for r in hkl_refs) * 1.1
   450|   450|            for rv in [max_r*0.25, max_r*0.5, max_r*0.75, max_r]:
   451|   451|                ax_arg.add_patch(plt.Circle((0,0), rv, fill=False, ls="--", lw=0.5, color="gray", alpha=0.3))
   452|   452|            ax_arg.set_xlim(-max_r, max_r); ax_arg.set_ylim(-max_r, max_r); ax_arg.set_aspect("equal")
   453|   453|            ax_arg.set_xlabel("Re(F)"); ax_arg.set_ylabel("Im(F)"); ax_arg.set_title("Argand-Diagramm F(hkl)")
   454|   454|            ax_arg.grid(True, alpha=0.3); fig_arg.tight_layout(); st.pyplot(fig_arg)
   455|   455|
   456|   456|            st.markdown("### 📊 d-spacing Abgleich")
   457|   457|            qual = []
   458|   458|            for m in matched:
   459|   459|                if m["h"] != "—":
   460|   460|                    for r in hkl_refs:
   461|   461|                        if r["h"]==m["h"] and r["k"]==m["k"] and r["l"]==m["l"]:
   462|   462|                            dd = abs(m.get("d (Å)",0) - r["d (Å)"])
   463|   463|                            qual.append({"hkl":f"{m['h']}{m['k']}{m['l']}","d_obs":m.get("d (Å)",0),
   464|   464|                                         "d_calc":r["d (Å)"],"Δd":round(dd,5),"Δd/d%":round(dd/r["d (Å)"]*100,3)})
   465|   465|                            break
   466|   466|            if qual:
   467|   467|                st.dataframe(qual, use_container_width=True, hide_index=True)
   468|   468|                dv = [q["Δd"] for q in qual]
   469|   469|                c1,c2,c3,c4 = st.columns(4)
   470|   470|                c1.metric("Reflexe",len(qual)); c2.metric("Mittl. Δd",f"{np.mean(dv):.5f} Å")
   471|   471|                c3.metric("RMS Δd",f"{math.sqrt(np.mean(np.array(dv)**2)):.5f} Å"); c4.metric("Max Δd",f"{max(dv):.5f} Å")
   472|   472|
   473|   473|            st.markdown("### 🔬 Vollständige HKL-Tabelle")
   474|   474|            fc = ["h","k","l","d (Å)","2θ (°)","|F|²","|F|","F_real","F_imag","φ (°)"]
   475|   475|            st.dataframe([{k:r[k] for k in fc} for r in hkl_refs], use_container_width=True, hide_index=True)
   476|   476|
   477|   477|            st.markdown("### 💾 Export")
   478|   478|            b1 = io.StringIO(); w1 = csv.DictWriter(b1,fieldnames=fc); w1.writeheader(); w1.writerows([{k:r[k] for k in fc} for r in hkl_refs])
   479|   479|            b1b = base64.b64encode(b1.getvalue().encode()).decode()
   480|   480|            st.markdown(f'<a href="data:text/csv;base64,{b1b}" download="hkl_reflections.csv">📥 CSV (alle Reflexe)</a>',unsafe_allow_html=True)
   481|   481|            mc = ["2θ obs","h","k","l","d (Å)","Δ2θ","|F|²","Intensität"]
   482|   482|            b2 = io.StringIO(); w2 = csv.DictWriter(b2,fieldnames=mc); w2.writeheader(); w2.writerows([{k:m.get(k,"—") for k in mc} for m in matched])
   483|   483|            b2b = base64.b64encode(b2.getvalue().encode()).decode()
   484|   484|            st.markdown(f'<a href="data:text/csv;base64,{b2b}" download="indexed_peaks.csv">📥 CSV (indizierte Peaks)</a>',unsafe_allow_html=True)
   485|   485|
   486|   486|# ==================== TAB 4 ====================
   487|   487|with tab4:
   488|   488|    st.markdown("2θ- und Intensitätswerte zeilenweise — Leerzeichen oder Tab getrennt.")
   489|   489|    manual = st.text_area("Daten", height=200, placeholder="10.5 120\n12.3 450\n14.1 230\n...")
   490|   490|    if st.button("Diagramm zeichnen") and manual.strip():
   491|   491|        res = parse_xy(manual)
   492|   492|        if res:
   493|   493|            tt, intens = res
   494|   494|            st.pyplot(plot_xy(tt, intens, title="Manuelles Diffraktogramm", color="#d62728"))
   495|   495|            with st.expander("📄 Rohdaten"):
   496|   496|                st.dataframe({"2θ (°)": tt, "Intensität": intens}, use_container_width=True)
   497|   497|        else:
   498|   498|            st.error("Keine gültigen Zahlenpaare.")
   499|   499|
   500|   500|# ==================== TAB 5 ====================
# ==================== TAB 6: Rietveld ====================
with tab6:
    st.markdown("### 🧪 Rietveld-Verfeinerung (vereinfacht)")
    st.markdown("Full-Pattern-Fitting: Startmodell aus CIF → berechne gesamtes Diffraktogramm → optimiere Parameter per Least-Squares.")
    st.caption("Für FullProf/GSAS: CIF + .xy separat verwenden. Hier: vereinfachtes Pseudo-Voigt-Profil + Background-Polynom.")
    
    # Parameter-Initialisierung mit besseren Defaults und Grenzen
    if "riet_params" not in st.session_state:
        st.session_state.riet_params = {
            'scale': 1000.0,
            'zshift': 0.0,
            'U': 0.05,
            'V': -0.01,
            'W': 0.05,
            'a': a,
            'b': b,
            'c': c,
            'bg': [np.mean(intens_raw)] + [0.0] * 5
        }
    
    p = st.session_state.riet_params
    
    # Verbessertes UI-Layout
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        riet_hkl = st.slider("hkl-Bereich", 1, 15, 7, key="riet_hkl")
        p['scale'] = st.number_input("Skalenfaktor", value=float(p['scale']), format="%.2f", min_value=0.1, max_value=10000.0, step=10.0)
    with col_r2:
        p['zshift'] = st.number_input("Zero-Shift (°)", value=float(p['zshift']), format="%.4f", min_value=-1.0, max_value=1.0, step=0.001)
        p['U'] = st.number_input("Caglioti U", value=float(p['U']), format="%.5f", min_value=0.0, max_value=1.0, step=0.001)
    with col_r3:
        p['V'] = st.number_input("Caglioti V", value=float(p['V']), format="%.5f", min_value=-1.0, max_value=1.0, step=0.001)
        p['W'] = st.number_input("Caglioti W", value=float(p['W']), format="%.5f", min_value=0.0, max_value=1.0, step=0.001)
    
    # Lattice Parameter verfeinern
    st.markdown("#### 🔧 Lattice Parameter Verfeinerung")
    col_lat1, col_lat2, col_lat3 = st.columns(3)
    with col_lat1:
        p['a'] = st.number_input("a (Å)", value=float(p['a']), format="%.5f", step=0.0001)
        refine_a = st.checkbox("a verfeinern", value=True)
    with col_lat2:
        p['b'] = st.number_input("b (Å)", value=float(p['b']), format="%.5f", step=0.0001)
        refine_b = st.checkbox("b verfeinern", value=True)
    with col_lat3:
        p['c'] = st.number_input("c (Å)", value=float(p['c']), format="%.5f", step=0.0001)
        refine_c = st.checkbox("c verfeinern", value=True)

    # Background-Ordnung
    bg_order = st.selectbox("Background-Polynom Ordnung", [0, 1, 2, 3, 4, 5, 6], index=3, key="riet_bg_order")
    
    # Background-Koeffizienten
    bg_str = ",".join(f"{v:.2f}" for v in p['bg'][:bg_order+1])
    bg_input = st.text_input(f"Background-Koeffizienten (B₀...B{bg_order})", value=bg_str)
    try:
        bg_vals = [float(x.strip()) for x in bg_input.split(",")]
        if len(bg_vals) == bg_order + 1:
            p['bg'] = bg_vals + [0.0] * (len(p['bg']) - len(bg_vals))
    except ValueError:
        st.error("Ungültige Background-Werte")

    st.markdown("#### 🔧 Verfeinerungseinstellungen")
    col_rf1, col_rf2, col_rf3 = st.columns(3)
    with col_rf1:
        ref_scale = st.checkbox("Skalenfaktor", value=True)
        ref_zshift = st.checkbox("Zero-Shift", value=True)
    with col_rf2:
        ref_U = st.checkbox("Caglioti U", value=True)
        ref_V = st.checkbox("Caglioti V", value=True)
        ref_W = st.checkbox("Caglioti W", value=True)
    with col_rf3:
        ref_bg = st.checkbox("Background", value=True)

    # Verbesserte Calculation Logik
    if st.button("🧪 Pattern berechnen & Fit starten", type="primary"):
        with st.spinner("Optimiere..."):
            # HKL-Reflexe berechnen
            hkl_refs_r = compute_structure_factors(
                crystal["atoms"], 
                p['a'], p['b'], p['c'], 
                alpha, beta, gamma, 
                wavelength, 
                (riet_hkl, riet_hkl, riet_hkl)
            )
            
            # Variablen Parameter definieren
            var_keys = []
            x0_var = []
            bounds = []
            
            if ref_scale:
                var_keys.append('scale')
                x0_var.append(p['scale'])
                bounds.append((0.1, 10000.0))
            if ref_zshift:
                var_keys.append('zshift')
                x0_var.append(p['zshift'])
                bounds.append((-1.0, 1.0))
            if ref_U:
                var_keys.append('U')
                x0_var.append(p['U'])
                bounds.append((0.0, 1.0))
            if ref_V:
                var_keys.append('V')
                x0_var.append(p['V'])
                bounds.append((-1.0, 1.0))
            if ref_W:
                var_keys.append('W')
                x0_var.append(p['W'])
                bounds.append((0.0, 1.0))
            if refine_a:
                var_keys.append('a')
                x0_var.append(p['a'])
                bounds.append((0.1, 20.0))
            if refine_b:
                var_keys.append('b')
                x0_var.append(p['b'])
                bounds.append((0.1, 20.0))
            if refine_c:
                var_keys.append('c')
                x0_var.append(p['c'])
                bounds.append((0.1, 20.0))
            if ref_bg:
                for i in range(bg_order + 1):
                    var_keys.append(f'bg{i}')
                    x0_var.append(p['bg'][i])
                    bounds.append((None, None))  # Keine Grenzen für Background-Koeffizienten

            # Verbesserte Zielfunktion
            def objective(x_var):
                # Map flat vector x_var back to the dictionary params
                params = p.copy()
                for i, key in enumerate(var_keys):
                    if key.startswith('bg'):
                        # Background keys are handled as bg_coeffs list
                        bg_idx = int(key[2:])
                        if 'bg' not in params or not isinstance(params['bg'], list):
                            params['bg'] = [0.0] * (bg_order + 1)
                        params['bg'][bg_idx] = x_var[i]
                    else:
                        params[key] = x_var[i]
                
                y_calc = riet_calc_pattern(params, np.array(tt_raw), hkl_refs_r, bg_order, wavelength, alpha, beta, gamma)
                y_obs = np.array(intens_raw)
                
                # Verbesserte Gewichtung
                w = 1.0 / np.sqrt(np.maximum(y_obs, 1.0))
                residuals = (y_obs - y_calc) * w
                
                # Regularisierung für Background-Koeffizienten
                bg_reg = 0.0
                if ref_bg:
                    for i in range(bg_order + 1):
                        bg_reg += 0.001 * params['bg'][i]**2
                
                return np.sum(residuals**2) + bg_reg

            # Optimize mit verbesserten Einstellungen
            if len(x0_var) > 0:
                try:
                    res = least_squares(
                        objective, 
                        x0_var, 
                        bounds=[b[0] if b[0] is not None else -np.inf for b in bounds],
                                [b[1] if b[1] is not None else np.inf for b in bounds],
                        method="trf",
                        max_nfev=500,
                        ftol=1e-6,
                        xtol=1e-6
                    )
                    
                    # Update session state
                    final_p = p.copy()
                    for i, key in enumerate(var_keys):
                        if key.startswith('bg'):
                            final_p['bg'][int(key[2:])] = res.x[i]
                        else:
                            final_p[key] = res.x[i]
                    
                    st.session_state.riet_params = final_p
                    st.session_state.riet_hkl_refs = hkl_refs_r
                    st.session_state.riet_y_calc = riet_calc_pattern(
                        final_p, 
                        np.array(tt_raw), 
                        hkl_refs_r, 
                        bg_order, 
                        wavelength, 
                        alpha, 
                        beta, 
                        gamma
                    )
                    st.session_state.riet_res = res
                    st.rerun()
                except Exception as e:
                    st.error(f"Fehler in der Verfeinerung: {str(e)}")
            else:
                st.warning("Keine Parameter zur Verfeinerung ausgewählt.")

    # Ergebnisse anzeigen
    if "riet_y_calc" in st.session_state:
        y_calc = st.session_state.riet_y_calc
        y_obs = np.array(intens_raw)
        residuals = y_obs - y_calc
        
        fig_r, (ax_r, ax_d) = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax_r.plot(tt_raw, y_obs, color="black", lw=0.8, label="Obs")
        ax_r.plot(tt_raw, y_calc, color="red", lw=0.8, label="Calc")
        ax_r.legend()
        ax_r.grid(True, alpha=0.3)
        ax_d.plot(tt_raw, residuals, color="gray", lw=0.6)
        ax_d.axhline(0, color="black", lw=0.5)
        ax_d.fill_between(tt_raw, residuals, 0, alpha=0.3, color="gray")
        ax_d.set_xlabel("2θ (°)")
        ax_d.set_ylabel("Differenz")
        fig_r.tight_layout()
        st.pyplot(fig_r)
        
        # Verbesserte R-Faktoren
        rp = np.sum(np.abs(residuals)) / np.sum(np.abs(y_obs)) * 100
        rwp = np.sqrt(np.sum(residuals**2) / np.sum(y_obs**2)) * 100
        chi_sq = np.sum((residuals**2) / np.maximum(y_obs, 1.0))
        chi_sq_red = chi_sq / (len(y_obs) - len(st.session_state.riet_res.x))
        
        col_rp, col_rwp, col_chi, col_chired = st.columns(4)
        col_rp.metric("Rp (%)", f"{rp:.2f}")
        col_rwp.metric("Rwp (%)", f"{rwp:.2f}")
        col_chi.metric("χ²", f"{chi_sq:.2f}")
        col_chired.metric("χ²_red", f"{chi_sq_red:.2f}")

st.caption("FellX4 — mit HKL-Suche & Strukturfaktoren 🚀🔷")
