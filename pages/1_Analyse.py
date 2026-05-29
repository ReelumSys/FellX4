import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import io
import csv
import base64

st.set_page_config(page_title="FellX4 — Analyse", layout="wide")

# ──────────────────────────────────────────────
#  Prüfen: Daten da?
# ──────────────────────────────────────────────
if "crystal" not in st.session_state or "tt_raw" not in st.session_state:
    st.warning("⚠️ Keine Daten geladen — bitte zuerst CIF + Diffraktogramm auf der Startseite hochladen.")
    if st.button("← Neue Daten laden"):
        st.switch_page("pages/home.py")
    st.stop()

crystal = st.session_state["crystal"]
a = st.session_state["a"]
b = st.session_state["b"]
c = st.session_state["c"]
alpha = st.session_state["alpha"]
beta = st.session_state["beta"]
gamma = st.session_state["gamma"]
sg = st.session_state["sg"]
wavelength = st.session_state.get("wavelength", 1.5406)
tt_raw = st.session_state["tt_raw"]
intens_raw = st.session_state["intens_raw"]
peaks = st.session_state.get("peaks", [])
cif_name = st.session_state.get("cif_file_name", "?")
xrd_name = st.session_state.get("xrd_file_name", "?")

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

def find_peaks(tt, intens, prominence=0.05):
    from scipy.signal import find_peaks as sp_find_peaks
    from scipy.ndimage import gaussian_filter1d
    arr = np.array(intens, dtype=float)
    if np.max(arr) == 0:
        return []
    smoothed = gaussian_filter1d(arr, sigma=1.5)
    abs_prominence = prominence * np.max(arr)
    if len(tt) > 1:
        step = np.median(np.diff(tt))
        distance = max(3, int(0.5 / step))
    else:
        distance = 3
    peaks_idx, _ = sp_find_peaks(smoothed, prominence=abs_prominence, distance=distance, width=1)
    return [(tt[i], intens[i]) for i in peaks_idx]

def d_spacing(h, k, l, a, b, c, alpha, beta, gamma):
    al = math.radians(alpha)
    be = math.radians(beta)
    ga = math.radians(gamma)
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    sa, sb, sg = math.sin(al), math.sin(be), math.sin(ga)
    vol = a * b * c * math.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
    a_star = b * c * sa / vol
    b_star = a * c * sb / vol
    c_star = a * b * sg / vol
    ca_star = (cb * cg - ca) / (sb * sg)
    cb_star = (ca * cg - cb) / (sa * sg)
    cg_star = (ca * cb - cg) / (sa * sb)
    d2 = (h**2 * a_star**2 + k**2 * b_star**2 + l**2 * c_star**2
          + 2 * h * k * a_star * b_star * cg_star
          + 2 * h * l * a_star * c_star * cb_star
          + 2 * k * l * b_star * c_star * ca_star)
    if d2 <= 0:
        return None
    return 1.0 / math.sqrt(d2)

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
    coeffs = SCATTERING_FACTORS.get(element.capitalize())
    if coeffs is None:
        return 0.0
    s2 = sin_theta_over_lambda**2
    f = coeffs[4]
    for a_i, b_i in coeffs[:4]:
        f += a_i * math.exp(-b_i * s2)
    return f

def compute_structure_factors(atoms, a, b, c, alpha, beta, gamma, wavelength,
                               hkl_ranges=(10, 10, 10), min_f_sq_frac=0.01):
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
                F_real = 0.0; F_imag = 0.0
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
                        "h": h, "k": k, "l": l, "d (Å)": round(d, 4),
                        "2θ (°)": round(two_theta, 4), "|F|²": round(F_sq, 2),
                        "|F|": round(math.sqrt(F_sq), 2), "F_real": round(F_real, 3),
                        "F_imag": round(F_imag, 3), "φ (°)": round(math.degrees(phase_rad), 1),
                    })
    if raw_results:
        max_f = max(r["|F|²"] for r in raw_results)
        threshold = min_f_sq_frac * max_f
        results = [r for r in raw_results if r["|F|²"] >= threshold]
    else:
        results = []
    results.sort(key=lambda r: r["d (Å)"], reverse=True)
    return results

def match_peaks_to_hkl(peaks, hkl_data, wavelength, tol=0.2):
    matched = []
    for tt_obs, intens in peaks:
        best = None; best_delta = float("inf")
        for ref in hkl_data:
            delta = abs(ref["2θ (°)"] - tt_obs)
            if delta < best_delta and delta < tol:
                best_delta = delta; best = ref
        matched.append({"2θ obs": round(tt_obs, 4), "Intensität": round(intens, 2),
                        "h": best["h"] if best else "—", "k": best["k"] if best else "—",
                        "l": best["l"] if best else "—", "d (Å)": best["d (Å)"] if best else "—",
                        "Δ2θ": round(best_delta, 4) if best else "—",
                        "|F|": best["|F|"] if best else "—"})
    return matched

def frac_to_cart(frac, a, b, c, alpha, beta, gamma):
    ca = math.cos(math.radians(alpha)); cb = math.cos(math.radians(beta))
    cg = math.cos(math.radians(gamma)); sg = math.sin(math.radians(gamma))
    ax_v = a; bx = b * cg; by = b * sg; cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz = c * math.sqrt(1 - cb**2 - ((ca - cb * cg) / sg)**2)
    x, y, z = frac
    return (x * ax_v + y * bx + z * cx, y * by + z * cy, z * cz)

def plot_unit_cell(atoms, a, b, c, alpha, beta, gamma):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    corners_frac = [(0,0,0), (1,0,0), (1,1,0), (0,1,0),
                    (0,0,1), (1,0,1), (1,1,1), (0,1,1)]
    corners = [frac_to_cart(c, a, b, c, alpha, beta, gamma) for c in corners_frac]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    idx = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in idx:
        ax.plot([corners[i][0], corners[j][0]], [corners[i][1], corners[j][1]],
                [corners[i][2], corners[j][2]], color="#1f77b4", lw=1.5)
    faces_idx = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)]
    face_verts = [[corners[i] for i in f] for f in faces_idx]
    poly = Poly3DCollection(face_verts, alpha=0.05, facecolor="#1f77b4", edgecolor="none")
    ax.add_collection3d(poly)
    colors = {"H":"#ffffff","C":"#222222","N":"#3050f8","O":"#ff0d0d","Na":"#ab5cf2",
              "Mg":"#8aff00","Al":"#bfa6a6","Si":"#f0c8a0","P":"#ff8000","S":"#ffff30",
              "Cl":"#1ff01f","K":"#8f40d4","Ca":"#3dff00","Ti":"#bfc2c7","Fe":"#e06633",
              "Ni":"#50f050","Cu":"#c88033","Zn":"#71d0a0","Sr":"#00ff00","Ba":"#00c900","Pb":"#575961"}
    for atom in atoms:
        try:
            fx = float(atom.get("fract_x", 0)); fy = float(atom.get("fract_y", 0))
            fz = float(atom.get("fract_z", 0)); occ = float(atom.get("occupancy", 1.0))
            match = re.match(r"([A-Za-z]+)", atom.get("type_symbol", "H"))
            el = match.group(1) if match else "H"
        except: continue
        pos = frac_to_cart((fx, fy, fz), a, b, c, alpha, beta, gamma)
        ax.scatter(*pos, c=colors.get(el.capitalize(), "#888888"),
                   s=80+40*(occ if occ<=1 else 1), edgecolors="black", linewidths=0.3, alpha=0.85, zorder=10)
    av = frac_to_cart((1.15,0,0), a, b, c, alpha, beta, gamma)
    bv = frac_to_cart((0,1.15,0), a, b, c, alpha, beta, gamma)
    cv = frac_to_cart((0,0,1.15), a, b, c, alpha, beta, gamma)
    ax.text(*av, "a", fontsize=12, fontweight="bold", color="#1f77b4")
    ax.text(*bv, "b", fontsize=12, fontweight="bold", color="#1f77b4")
    ax.text(*cv, "c", fontsize=12, fontweight="bold", color="#1f77b4")
    mx = np.max(np.abs(np.array(corners)))
    ax.set_xlim(-mx*0.1, mx*1.1); ax.set_ylim(-mx*0.1, mx*1.1); ax.set_zlim(-mx*0.1, mx*1.1)
    ax.set_box_aspect([1,1,1]); ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    ax.set_title("Elementarzelle", fontsize=12, fontweight="bold"); fig.tight_layout()
    return fig

# ─── Rietveld helpers (defined once, reused across reruns) ───
from scipy.optimize import least_squares

def pseudo_voigt(x, mu, fwhm, eta=0.5):
    s = fwhm / (2 * math.sqrt(2 * math.log(2)))
    g = np.exp(-0.5 * ((x - mu) / s)**2) / (s * math.sqrt(2 * math.pi))
    l = fwhm / (2 * math.pi) / ((x - mu)**2 + (fwhm / 2)**2)
    return eta * l + (1 - eta) * g

def caglioti_fwhm(theta_deg, U, V, W):
    t = math.tan(math.radians(theta_deg))
    return math.sqrt(max(U * t**2 + V * t + W, 0.001))

def riet_calc_pattern(params, tt_obs, hkl_refs, bg_order, ref_a, ref_b, ref_c,
                       fix_a, fix_b, fix_c, wavelength, alpha, beta, gamma):
    idx = 0
    scale = params[idx]; idx += 1
    zshift = params[idx]; idx += 1
    U = params[idx]; idx += 1
    V = params[idx]; idx += 1
    W = params[idx]; idx += 1
    a_p = fix_a; b_p = fix_b; c_p = fix_c
    if ref_a: a_p = params[idx]; idx += 1
    if ref_b: b_p = params[idx]; idx += 1
    if ref_c: c_p = params[idx]; idx += 1
    bg_p = params[idx:idx + bg_order + 1]

    hkl_curr = []
    for r in hkl_refs:
        d = d_spacing(r["h"], r["k"], r["l"], a_p, b_p, c_p, alpha, beta, gamma)
        if d is None or d <= 0: continue
        stheta = wavelength / (2 * d)
        if abs(stheta) > 1: continue
        tt = 2 * math.degrees(math.asin(stheta))
        if tt < 5 or tt > 150: continue
        hkl_curr.append({"2θ": tt, "|F|²": r["|F|²"]})

    y_calc = np.zeros_like(tt_obs, dtype=float)
    for r in hkl_curr:
        tt_pos = r["2θ"] + zshift
        fwhm_val = caglioti_fwhm(tt_pos, U, V, W)
        y_calc += r["|F|²"] * scale * pseudo_voigt(tt_obs, tt_pos, fwhm_val)

    x_norm = (tt_obs - np.min(tt_obs)) / (np.max(tt_obs) - np.min(tt_obs) + 1e-10)
    bg = np.zeros_like(tt_obs)
    for i, b in enumerate(bg_p):
        bg += b * x_norm**i
    return y_calc + np.maximum(bg, 0)

def riet_residuals(params, tt_obs, y_obs, hkl_refs, bg_order, ref_a, ref_b, ref_c,
                    fix_a, fix_b, fix_c, wavelength, alpha, beta, gamma):
    y_calc = riet_calc_pattern(params, tt_obs, hkl_refs, bg_order, ref_a, ref_b, ref_c,
                                fix_a, fix_b, fix_c, wavelength, alpha, beta, gamma)
    w = np.where(y_obs > 0, 1.0 / np.sqrt(y_obs + 1), 1.0)
    return (y_obs - y_calc) * w

# ──────────────────────────────────────────────
#  Top Info + Navigation
# ──────────────────────────────────────────────
st.title("🔬 FellX4 — Analyse")
col_info, col_back = st.columns([3, 1])
with col_info:
    st.markdown(f"**CIF:** {cif_name}  ·  **XRD:** {xrd_name}  ·  "
                f"a={a:.3f} b={b:.3f} c={c:.3f}  ·  "
                f"λ={wavelength:.4f} Å  ·  {len(peaks)} Peaks")
with col_back:
    if st.button("← Neue Daten laden"):
        st.switch_page("pages/home.py")

# ──────────────────────────────────────────────
#  TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁 Diffraktogramm",
    "📊 Vergleich + Subtraktion",
    "🔷 HKL + Strukturfaktoren",
    "✏️ Manuelle Eingabe",
    "📐 FWHM + Scherrer",
    "🧪 Rietveld",
])

# ==================== TAB 1 ====================
with tab1:
    st.pyplot(plot_xy(tt_raw, intens_raw, title=xrd_name))
    with st.expander("📄 Rohdaten"):
        st.dataframe({"2θ (°)": tt_raw, "Intensität": intens_raw}, use_container_width=True)

# ==================== TAB 2 ====================
with tab2:
    col_a, col_b = st.columns(2)
    with col_a: fa = st.file_uploader("Datei A", type=["xy", "txt", "csv"], key="c2a")
    with col_b: fb = st.file_uploader("Datei B", type=["xy", "txt", "csv"], key="c2b")
    if fa and fb:
        da = parse_xy(fa.read().decode("utf-8")); db = parse_xy(fb.read().decode("utf-8"))
        if da and db:
            x1, y1 = da; x2, y2 = db
            if len(x1) >= len(x2):
                y2i = np.interp(x1, x2, y2); diff = np.array(y1) - y2i; xd = x1
            else:
                y1i = np.interp(x2, x1, y1); diff = np.array(y1i) - y2; xd = x2
            fig_cmp, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                                gridspec_kw={"height_ratios": [2, 1]})
            ax1.plot(x1, y1, color="#1f77b4", lw=1.0, label=f"A: {fa.name}")
            ax1.plot(x2, y2, color="#d62728", lw=1.0, label=f"B: {fb.name}")
            ax1.set_ylabel("Intensität"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
            ax2.axhline(0, color="gray", lw=0.8, ls="--")
            ax2.plot(xd, diff, color="#2ca02c", lw=1.0)
            ax2.set_xlabel("2θ (°)"); ax2.set_ylabel("A − B"); ax2.grid(True, alpha=0.3)
            fig_cmp.tight_layout(); st.pyplot(fig_cmp)
    elif not fa and not fb:
        st.info("Lade beide Dateien für den Vergleich.")

# ==================== TAB 3 ====================
with tab3:
    st.markdown("### 📄 CIF-Struktur")
    st.success(f"✅ {cif_name} — a={a:.3f} b={b:.3f} c={c:.3f} Å, "
               f"α={alpha:.2f} β={beta:.2f} γ={gamma:.2f}°, {sg} | {len(crystal['atoms'])} Atome")
    with st.expander("📄 Atompositionen"):
        st.dataframe(crystal["atoms"], use_container_width=True)
    if st.checkbox("🔬 3D-Elementarzelle anzeigen", value=True):
        st.pyplot(plot_unit_cell(crystal["atoms"], a, b, c, alpha, beta, gamma))

    st.markdown("---"); st.markdown("### 📊 Diffraktogramm (experimentell)")
    prominence = st.slider("Peak-Empfindlichkeit", 0.0, 0.5, 0.05, 0.01, key="hkl_prom")
    local_peaks = find_peaks(tt_raw, intens_raw, prominence=prominence)
    fig_p, (ax_p, ax_peaks) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 1]})
    ax_p.plot(tt_raw, intens_raw, color="#1f77b4", lw=1.0)
    pt = [p[0] for p in local_peaks]; pi = [p[1] for p in local_peaks]
    ax_p.scatter(pt, pi, color="red", s=30, zorder=5, label=f"{len(local_peaks)} Peaks")
    ax_p.set_ylabel("Intensität"); ax_p.legend(fontsize=9); ax_p.grid(True, alpha=0.3)
    ax_peaks.bar(pt, pi, width=0.08, color="red", alpha=0.6)
    ax_peaks.set_xlabel("2θ (°)"); ax_peaks.grid(True, alpha=0.3)
    fig_p.tight_layout(); st.pyplot(fig_p)
    st.info(f"{len(local_peaks)} Peaks detektiert")

    hkl_range = st.slider("hkl-Suchbereich", 1, 10, 5, key="hkl_range")
    tol = st.slider("Toleranz Δ2θ (°)", 0.05, 1.0, 0.3, 0.05, key="hkl_tol")
    st.markdown("---")

    if st.button("🧮 hkl-Indizierung + Strukturfaktoren berechnen", type="primary"):
        if not local_peaks:
            st.error("❌ Keine Peaks — Empfindlichkeit erhöhen.")
        else:
            with st.spinner("Berechne hkl-Reflexe..."):
                hkl_refs = compute_structure_factors(
                    crystal["atoms"], a, b, c, alpha, beta, gamma,
                    wavelength, (hkl_range, hkl_range, hkl_range))
            st.success(f"✅ {len(hkl_refs)} theoretische Reflexe berechnet.")
            matched = match_peaks_to_hkl(local_peaks, hkl_refs, wavelength, tol=tol)
            matched_count = sum(1 for m in matched if m["h"] != "—")
            st.session_state["hkl_refs"] = hkl_refs
            st.session_state["matched"] = matched
            st.session_state["local_peaks"] = local_peaks

            st.markdown("### 📈 Annotiertes Diffraktogramm")
            fig_ann, ax_ann = plt.subplots(figsize=(12, 5))
            ax_ann.plot(tt_raw, intens_raw, color="#1f77b4", lw=1.0, label="Experiment")
            ref_tt = [r["2θ (°)"] for r in hkl_refs]
            ref_f = [r["|F|²"] for r in hkl_refs]
            if ref_f:
                ax_ann.vlines(ref_tt, 0, np.array(ref_f)/max(ref_f)*100, color="#1f77b4", lw=1.5, alpha=0.6, label="Berechnet")
            for m in matched:
                if m["h"] != "—":
                    ax_ann.scatter(m["2θ obs"], m["Intensität"], color="red", s=40, zorder=5)
                    ax_ann.annotate(f"{m['h']}{m['k']}{m['l']}", (m["2θ obs"], m["Intensität"]),
                                    textcoords="offset points", xytext=(0, 12), fontsize=7, ha="center", rotation=90,
                                    color="darkred", fontweight="bold")
            ax_ann.set_xlabel("2θ (°)"); ax_ann.set_ylabel("Intensität")
            ax_ann.set_title("Diffraktogramm mit hkl-Zuordnung", fontsize=12)
            ax_ann.legend(fontsize=9); ax_ann.grid(True, alpha=0.3); fig_ann.tight_layout(); st.pyplot(fig_ann)
            st.info(f"{matched_count}/{len(matched)} Peaks zugeordnet ({len(hkl_refs)} berechnete Reflexe)")

            st.markdown("### 📋 Indizierte Peaks & F(hkl)")
            disp = ["2θ obs", "h", "k", "l", "d (Å)", "Δ2θ", "|F|²", "|F|", "Intensität"]
            st.dataframe([{k: m[k] for k in disp if k in m} for m in matched], use_container_width=True, hide_index=True)

            st.markdown("### 🌀 Argand-Diagramm")
            fig_arg, ax_arg = plt.subplots(figsize=(6, 6))
            ax_arg.axhline(0, color="gray", lw=0.8); ax_arg.axvline(0, color="gray", lw=0.8)
            mh = {(m["h"], m["k"], m["l"]) for m in matched if m["h"] != "—"}
            for r in hkl_refs:
                key = (r["h"], r["k"], r["l"]); flag = key in mh
                ax_arg.scatter(r["F_real"], r["F_imag"], c="red" if flag else "#1f77b4",
                               s=60 if flag else 20, alpha=1.0 if flag else 0.4, zorder=5)
                if flag: ax_arg.annotate(f"{r['h']}{r['k']}{r['l']}", (r["F_real"], r["F_imag"]), fontsize=6, ha="center", va="bottom")
            max_r = max(math.sqrt(r["F_real"]**2 + r["F_imag"]**2) for r in hkl_refs) * 1.1
            for rv in [max_r*0.25, max_r*0.5, max_r*0.75, max_r]:
                ax_arg.add_patch(plt.Circle((0,0), rv, fill=False, ls="--", lw=0.5, color="gray", alpha=0.3))
            ax_arg.set_xlim(-max_r, max_r); ax_arg.set_ylim(-max_r, max_r); ax_arg.set_aspect("equal")
            ax_arg.set_xlabel("Re(F)"); ax_arg.set_ylabel("Im(F)"); ax_arg.set_title("Argand-Diagramm F(hkl)")
            ax_arg.grid(True, alpha=0.3); fig_arg.tight_layout(); st.pyplot(fig_arg)

            st.markdown("### 📊 d-spacing Abgleich")
            qual = []
            for m in matched:
                if m["h"] != "—":
                    for r in hkl_refs:
                        if r["h"]==m["h"] and r["k"]==m["k"] and r["l"]==m["l"]:
                            dd = abs(m.get("d (Å)",0) - r["d (Å)"])
                            qual.append({"hkl":f"{m['h']}{m['k']}{m['l']}","d_obs":m.get("d (Å)",0),
                                         "d_calc":r["d (Å)"],"Δd":round(dd,5),"Δd/d%":round(dd/r["d (Å)"]*100,3)})
                            break
            if qual:
                st.dataframe(qual, use_container_width=True, hide_index=True)
                dv = [q["Δd"] for q in qual]
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Reflexe",len(qual)); c2.metric("Mittl. Δd",f"{np.mean(dv):.5f} Å")
                c3.metric("RMS Δd",f"{math.sqrt(np.mean(np.array(dv)**2)):.5f} Å"); c4.metric("Max Δd",f"{max(dv):.5f} Å")

            st.markdown("### 🔬 Vollständige HKL-Tabelle")
            fc = ["h","k","l","d (Å)","2θ (°)","|F|²","|F|","F_real","F_imag","φ (°)"]
            st.dataframe([{k:r[k] for k in fc} for r in hkl_refs], use_container_width=True, hide_index=True)

            st.markdown("### 💾 Export")
            b1 = io.StringIO(); w1 = csv.DictWriter(b1,fieldnames=fc); w1.writeheader(); w1.writerows([{k:r[k] for k in fc} for r in hkl_refs])
            b1b = base64.b64encode(b1.getvalue().encode()).decode()
            st.markdown(f'<a href="data:text/csv;base64,{b1b}" download="hkl_reflections.csv">📥 CSV (alle Reflexe)</a>',unsafe_allow_html=True)
            mc = ["2θ obs","h","k","l","d (Å)","Δ2θ","|F|²","Intensität"]
            b2 = io.StringIO(); w2 = csv.DictWriter(b2,fieldnames=mc); w2.writeheader(); w2.writerows([{k:m.get(k,"—") for k in mc} for m in matched])
            b2b = base64.b64encode(b2.getvalue().encode()).decode()
            st.markdown(f'<a href="data:text/csv;base64,{b2b}" download="indexed_peaks.csv">📥 CSV (indizierte Peaks)</a>',unsafe_allow_html=True)

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

# ==================== TAB 5 ====================
with tab5:
    st.markdown("### 📐 FWHM-Diffraktogramm-Analyzer")
    fwhm_wl = st.number_input("λ (Å)", value=wavelength, format="%.4f")
    fwhm_prom = st.slider("Peak-Empfindlichkeit", 0.0, 0.5, 0.05, 0.01, key="fwhm_prom")
    scherrer_K = st.number_input("Scherrer-K (Formfaktor)", value=0.9, format="%.2f")

    tt_fwhm, intens_fwhm = tt_raw, intens_raw
    fwhm_peaks = find_peaks(tt_fwhm, intens_fwhm, prominence=fwhm_prom)

    if tt_fwhm is not None and intens_fwhm is not None:
        if len(fwhm_peaks) < 2:
            st.warning("Weniger als 2 Peaks — Empfindlichkeit erhöhen.")
        else:
            from scipy.optimize import curve_fit
            def gauss(x, A, mu, sigma, bg):
                return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + bg

            fwhm_results = []; fit_curves = []
            tt_arr = np.array(tt_fwhm); intens_arr = np.array(intens_fwhm)
            peaks_sorted = sorted(fwhm_peaks, key=lambda p: p[0])

            for idx, (p_tt, p_int) in enumerate(peaks_sorted):
                hw = 2.0
                if idx > 0: hw = min(hw, (p_tt - peaks_sorted[idx-1][0]) * 0.6)
                if idx < len(peaks_sorted)-1: hw = min(hw, (peaks_sorted[idx+1][0] - p_tt) * 0.6)
                mask = (tt_arr >= p_tt - hw) & (tt_arr <= p_tt + hw)
                xd, yd = tt_arr[mask], intens_arr[mask]
                if len(xd) < 5: continue
                try:
                    bg0, A0, sigma0 = np.min(yd), np.max(yd) - np.min(yd), 0.1
                    popt, _ = curve_fit(gauss, xd, yd, p0=[A0, p_tt, sigma0, bg0], maxfev=2000)
                    A_f, mu_f, sigma_f, bg_f = popt
                    if sigma_f <= 0 or A_f <= 0: continue
                    fwhm = 2 * math.sqrt(2 * math.log(2)) * sigma_f
                    fwhm_rad = math.radians(fwhm)
                    theta_r = math.radians(mu_f / 2)
                    D = scherrer_K * fwhm_wl / (fwhm_rad * math.cos(theta_r)) if fwhm_rad > 0 else 0
                    res = yd - gauss(xd, *popt)
                    ss_res = np.sum(res**2); ss_tot = np.sum((yd - np.mean(yd))**2)
                    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0
                    fwhm_results.append({"Peak":idx+1,"2θ (°)":round(mu_f,4),"Intensität":round(A_f+bg_f,1),
                                         "FWHM (°)":round(fwhm,4),"FWHM (rad)":round(fwhm_rad,6),
                                         "σ":round(sigma_f,4),"R²":round(r2,4),"D (nm)":round(D,2) if D>0 else 0})
                    fit_curves.append((xd, gauss(xd, *popt), mu_f))
                except (RuntimeError, ValueError): continue

            if not fwhm_results:
                st.error("Keine Peaks gefittet.")
            else:
                st.success(f"**{len(fwhm_results)} Peaks gefittet**")
                fv = [r["FWHM (°)"] for r in fwhm_results]
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Peaks",len(fwhm_results)); c2.metric("Mittl. FWHM",f"{np.mean(fv):.4f}°")
                c3.metric("Min FWHM",f"{min(fv):.4f}°"); c4.metric("Max FWHM",f"{max(fv):.4f}°")

                st.markdown("### 📈 Diffraktogramm + Fits")
                fig_fit, ax_fit = plt.subplots(figsize=(12,5))
                ax_fit.plot(tt_fwhm, intens_fwhm, color="#1f77b4", lw=0.8, label="Experiment", alpha=0.7)
                for xf, yf, mu in fit_curves:
                    ax_fit.plot(xf, yf, color="red", lw=1.5); ax_fit.axvline(mu, color="red", ls="--", lw=0.5, alpha=0.4)
                ax_fit.set_xlabel("2θ (°)"); ax_fit.set_ylabel("Intensität"); ax_fit.legend(fontsize=8); ax_fit.grid(True, alpha=0.3)
                fig_fit.tight_layout(); st.pyplot(fig_fit)

                fwhm_cols = ["Peak","2θ (°)","Intensität","FWHM (°)","FWHM (rad)","σ","R²","D (nm)"]
                fwhm_df = [{k:r[k] for k in fwhm_cols} for r in fwhm_results]
                st.dataframe(fwhm_df, use_container_width=True, hide_index=True)

                st.markdown("### 📉 Scherrer & Williamson-Hall")
                wh_data = [r for r in fwhm_results if r["D (nm)"] > 0]
                if wh_data:
                    theta_arr = np.array([math.radians(d["2θ (°)"]/2) for d in wh_data])
                    beta_arr = np.array([d["FWHM (rad)"] for d in wh_data])
                    x_wh = 4*np.sin(theta_arr); y_wh = beta_arr*np.cos(theta_arr)
                    coeffs = np.polyfit(x_wh, y_wh, 1); slope, intercept = coeffs
                    d_wh = scherrer_K*fwhm_wl/intercept if intercept>0 else 0
                    strain_val = slope; x_fit = np.linspace(min(x_wh)*0.8, max(x_wh)*1.2, 50); y_fit = np.polyval(coeffs, x_fit)

                    col_d, col_f = st.columns(2)
                    with col_d:
                        fig_d, ax_d = plt.subplots(figsize=(5,4))
                        ax_d.scatter([d["2θ (°)"] for d in wh_data], [d["D (nm)"] for d in wh_data], color="red", s=60, zorder=5)
                        md = np.mean([d["D (nm)"] for d in wh_data])
                        ax_d.axhline(md, color="gray", ls="--", lw=1, label=f"Mittel: {md:.1f} nm")
                        ax_d.set_xlabel("2θ (°)"); ax_d.set_ylabel("D (nm)"); ax_d.legend(fontsize=8); ax_d.grid(True, alpha=0.3)
                        fig_d.tight_layout(); st.pyplot(fig_d)
                    with col_f:
                        fig_f2, ax_f2 = plt.subplots(figsize=(5,4))
                        ax_f2.scatter([d["2θ (°)"] for d in wh_data], [d["FWHM (°)"] for d in wh_data], color="darkorange", s=60, zorder=5)
                        ax_f2.set_xlabel("2θ (°)"); ax_f2.set_ylabel("FWHM (°)"); ax_f2.grid(True, alpha=0.3)
                        fig_f2.tight_layout(); st.pyplot(fig_f2)

                    fig_wh, ax_wh = plt.subplots(figsize=(7,5))
                    ax_wh.scatter(x_wh, y_wh, color="darkgreen", s=70, zorder=5, label="Daten")
                    ax_wh.plot(x_fit, y_fit, color="red", lw=1.5, label=f"Fit: y={slope:.4f}x+{intercept:.4f}")
                    ax_wh.set_xlabel("4 sinθ"); ax_wh.set_ylabel("β cosθ (rad)"); ax_wh.set_title("Williamson-Hall Plot")
                    ax_wh.legend(fontsize=9); ax_wh.grid(True, alpha=0.3); fig_wh.tight_layout(); st.pyplot(fig_wh)

                    da = [d["D (nm)"] for d in wh_data]
                    c1s,c2s,c3s,_ = st.columns(4)
                    c1s.metric("Mittl. D (Scherrer)",f"{np.mean(da):.1f} nm"); c2s.metric("Min D",f"{min(da):.1f} nm")
                    c3s.metric("Max D",f"{max(da):.1f} nm")
                    c1w,c2w,c3w,c4w = st.columns(4)
                    c1w.metric("D (W-H)",f"{d_wh:.1f} nm" if d_wh>0 else "—"); c2w.metric("Mikrodehnung ε",f"{strain_val:.6f}")
                    c3w.metric("R² (W-H)",f"{np.corrcoef(x_wh,y_wh)[0,1]**2:.4f}"); c4w.metric("Anzahl",len(wh_data))
                else:
                    st.info("Keine Kristallitgrößen berechnet.")

                for row_start in range(0, len(fwhm_results), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        pi = row_start + j
                        if pi >= len(fwhm_results): break
                        r = fwhm_results[pi]
                        with cols[j]:
                            mu = r["2θ (°)"]
                            mask = (tt_arr >= mu-2) & (tt_arr <= mu+2)
                            xw, yw = tt_arr[mask], intens_arr[mask]
                            if len(xw) < 3: continue
                            fig_pi, ax_pi = plt.subplots(figsize=(5,3))
                            ax_pi.plot(xw, yw, color="#1f77b4", lw=1, label="Data")
                            mask2 = (tt_arr >= mu-1.5) & (tt_arr <= mu+1.5)
                            xw2, yw2 = tt_arr[mask2], intens_arr[mask2]
                            if len(xw2) >= 5:
                                try:
                                    popt2, _ = curve_fit(gauss, xw2, yw2, p0=[r["Intensität"], mu, 0.1, np.min(yw2)], maxfev=2000)
                                    ax_pi.plot(xw2, gauss(xw2, *popt2), "red", lw=1.5, label="Fit")
                                except: pass
                            ax_pi.axvline(mu, color="red", ls="--", lw=0.8, alpha=0.5)
                            ax_pi.set_title(f"Peak {r['Peak']}: {mu:.2f}°  FWHM={r['FWHM (°)']:.4f}°  D={r['D (nm)']:.1f}nm", fontsize=9)
                            ax_pi.set_xlabel("2θ (°)",fontsize=8); ax_pi.set_ylabel("Intensität",fontsize=8)
                            ax_pi.tick_params(labelsize=7); ax_pi.grid(True, alpha=0.3); fig_pi.tight_layout(); st.pyplot(fig_pi)

                buf = io.StringIO()
                w = csv.DictWriter(buf,fieldnames=fwhm_cols); w.writeheader(); w.writerows(fwhm_df)
                b64 = base64.b64encode(buf.getvalue().encode()).decode()
                st.markdown(f'<a href="data:text/csv;base64,{b64}" download="fwhm_results.csv">📥 FWHM-Ergebnisse als CSV</a>',unsafe_allow_html=True)

# ==================== TAB 6: Rietveld ====================
with tab6:
    st.markdown("### 🧪 Rietveld-Verfeinerung (vereinfacht)")
    st.markdown("Full-Pattern-Fitting: Startmodell aus CIF → berechne gesamtes Diffraktogramm → optimiere Parameter per Least-Squares.")
    st.caption("Für FullProf/GSAS: CIF + .xy separat verwenden. Hier: vereinfachtes Pseudo-Voigt-Profil + Background-Polynom.")

    # ─── Parameter UI always visible ───
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        riet_hkl = st.slider("hkl-Bereich", 1, 10, 5, key="riet_hkl")
        riet_scale = st.number_input("Skalenfaktor", value=1.0, format="%.4f", key="riet_scale")
    with col_r2:
        riet_zshift = st.number_input("Zero-Shift (°)", value=0.0, format="%.4f", key="riet_zshift")
        riet_U = st.number_input("Caglioti U", value=0.01, format="%.5f", key="riet_U")
    with col_r3:
        riet_V = st.number_input("Caglioti V", value=-0.005, format="%.5f", key="riet_V")
        riet_W = st.number_input("Caglioti W", value=0.005, format="%.5f", key="riet_W")

    bg_order = st.selectbox("Background-Polynom Ordnung", [0,1,2,3,4,5], index=3, key="riet_bg_order")
    bg_def = [sum(intens_raw)/len(intens_raw)*0.5] + [0.0]*bg_order
    bg_input = st.text_input(f"Background-Koeffizienten (B₀...B{bg_order}, kommagetrennt)",
                              value=",".join(f"{v:.2f}" for v in bg_def), key="riet_bg")
    bg_coeffs = None
    try:
        bg_coeffs = [float(x.strip()) for x in bg_input.split(",")]
        if len(bg_coeffs) != bg_order + 1:
            st.error(f"Brauche {bg_order + 1} Koeffizienten, habe {len(bg_coeffs)}")
            bg_coeffs = None
    except ValueError:
        st.error("Ungültige Zahlen. Kommagetrennt eingeben (z.B. 150.0,0.0,0.0,0.0)")

    st.markdown("#### 🔧 Verfeinerungsparameter")
    col_rf1, col_rf2, col_rf3 = st.columns(3)
    with col_rf1:
        ref_scale = st.checkbox("Skalenfaktor", value=True)
        ref_a = st.checkbox("a", value=False)
        ref_b = st.checkbox("b", value=False)
        ref_c = st.checkbox("c", value=False)
    with col_rf2:
        ref_zshift = st.checkbox("Zero-Shift", value=True)
        ref_U = st.checkbox("Caglioti U", value=True)
        ref_V = st.checkbox("Caglioti V", value=True)
        ref_W = st.checkbox("Caglioti W", value=True)
    with col_rf3:
        ref_bg = st.checkbox("Background", value=True)

    # ─── Step 1: Volles Pattern berechnen ───
    if st.button("🧪 Volles Pattern berechnen", type="primary", key="riet_calc"):
        if bg_coeffs is None:
            st.error("Background-Koeffizienten korrigieren.")
        else:
            with st.spinner("Berechne Reflexe und generiere Pattern..."):
                hkl_refs_r = compute_structure_factors(
                    crystal["atoms"], a, b, c, alpha, beta, gamma,
                    wavelength, (riet_hkl, riet_hkl, riet_hkl))
            if not hkl_refs_r:
                st.error("Keine Reflexe berechnet.")
            else:
                # Build initial param vector
                x0 = [riet_scale, riet_zshift, riet_U, riet_V, riet_W]
                if ref_a: x0.append(a)
                if ref_b: x0.append(b)
                if ref_c: x0.append(c)
                x0 += bg_coeffs

                y_init = riet_calc_pattern(x0, np.array(tt_raw), hkl_refs_r, bg_order,
                                           ref_a, ref_b, ref_c, a, b, c,
                                           wavelength, alpha, beta, gamma)
                res_init = np.array(intens_raw) - y_init
                chi2_init = np.sum(res_init**2 / (np.array(intens_raw) + 1))

                st.session_state["riet_hkl_refs"] = hkl_refs_r
                st.session_state["riet_x0"] = x0
                st.session_state["riet_y_init"] = y_init
                st.session_state["riet_chi2_init"] = chi2_init
                st.session_state["riet_ref_a"] = ref_a
                st.session_state["riet_ref_b"] = ref_b
                st.session_state["riet_ref_c"] = ref_c
                st.session_state["riet_bg_order"] = bg_order
                st.rerun()

    # ─── Step 2: Show initial results (if calculated) ───
    if st.session_state.get("riet_hkl_refs") is not None:
        hkl_refs_r = st.session_state["riet_hkl_refs"]
        x0 = st.session_state["riet_x0"]
        y_init = st.session_state["riet_y_init"]
        chi2_init = st.session_state["riet_chi2_init"]
        raff_a = st.session_state.get("riet_ref_a", False)
        raff_b = st.session_state.get("riet_ref_b", False)
        raff_c = st.session_state.get("riet_ref_c", False)
        bg_ord = st.session_state.get("riet_bg_order", 3)

        y_obs = np.array(intens_raw)
        res_init = y_obs - y_init
        s_res = np.sum(res_init**2); s_obs = np.sum(y_obs**2)
        rp_init = np.sum(np.abs(res_init)) / np.sum(np.abs(y_obs)) * 100
        rwp_init = math.sqrt(s_res / s_obs) * 100

        # ─── Difference Plot (initial) ───
        st.markdown("### 📈 Initiales Pattern")
        fig_ri, (ax_ri, ax_diff) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                                                 gridspec_kw={"height_ratios": [3, 1]})
        ax_ri.plot(tt_raw, intens_raw, color="black", lw=0.8, label="I_obs")
        ax_ri.plot(tt_raw, y_init, color="red", lw=0.8, label="I_calc (initial)")
        for r in hkl_refs_r:
            ax_ri.axvline(r["2θ (°)"], color="green", lw=0.3, alpha=0.3, ls="--")
        ax_ri.set_ylabel("Intensität"); ax_ri.legend(fontsize=9); ax_ri.grid(True, alpha=0.3)
        ax_ri.set_title(f"Rietveld-Plot (initial) — χ² = {chi2_init:.1f}", fontsize=11)
        ax_diff.plot(tt_raw, res_init, color="gray", lw=0.6)
        ax_diff.axhline(0, color="black", lw=0.5)
        ax_diff.fill_between(tt_raw, res_init, 0, alpha=0.3, color="gray")
        ax_diff.set_xlabel("2θ (°)"); ax_diff.set_ylabel("I_obs − I_calc"); ax_diff.grid(True, alpha=0.3)
        fig_ri.tight_layout(); st.pyplot(fig_ri)

        col_rp1, col_rp2, col_rp3 = st.columns(3)
        col_rp1.metric("Rₚ (%)", f"{rp_init:.2f}")
        col_rp2.metric("R_wₚ (%)", f"{rwp_init:.2f}")
        col_rp3.metric("χ²", f"{chi2_init:.1f}")

        # ─── Step 3: Refinement button (top-level, NOT nested) ───
        n_params = 5 + (1 if raff_a else 0) + (1 if raff_b else 0) + (1 if raff_c else 0) + (bg_ord + 1)
        varying_count = 5 + (1 if raff_a else 0) + (1 if raff_b else 0) + (1 if raff_c else 0)
        if ref_bg: varying_count += bg_ord + 1
        if varying_count >= len(tt_raw):
            st.error(f"Zu viele Parameter ({varying_count}) für {len(tt_raw)} Datenpunkte. Weniger Parameter auswählen oder hkl-Bereich erhöhen.")
        else:
            if st.button("⚡ Verfeinerung starten (Least-Squares)", key="riet_refine"):
                with st.spinner("Optimiere Parameter..."):
                    # Build varying mask
                    varying = [True] * 5
                    if raff_a: varying.append(True)
                    else: varying.append(False)
                    if raff_b: varying.append(True)
                    else: varying.append(False)
                    if raff_c: varying.append(True)
                    else: varying.append(False)
                    varying += [ref_bg] * (bg_ord + 1)

                    x0_var = [x0[i] for i, v in enumerate(varying) if v]

                    # Bounds
                    bounds_low = [0.001, -1.0, 0.0, -0.1, 0.0]
                    bounds_high = [1000, 1.0, 1.0, 0.1, 1.0]
                    if raff_a: bounds_low.append(a*0.9); bounds_high.append(a*1.1)
                    if raff_b: bounds_low.append(b*0.9); bounds_high.append(b*1.1)
                    if raff_c: bounds_low.append(c*0.9); bounds_high.append(c*1.1)
                    bounds_low += [0.0] + [-1e6]*bg_ord
                    bounds_high += [1e10] + [1e6]*bg_ord
                    low_var = [bounds_low[i] for i, v in enumerate(varying) if v]
                    high_var = [bounds_high[i] for i, v in enumerate(varying) if v]

                    def resid_var(x_var):
                        x_full = list(x0)
                        j = 0
                        for i in range(len(varying)):
                            if varying[i]: x_full[i] = x_var[j]; j += 1
                        return riet_residuals(x_full, np.array(tt_raw), np.array(intens_raw),
                                              hkl_refs_r, bg_ord, raff_a, raff_b, raff_c,
                                              a, b, c, wavelength, alpha, beta, gamma)

                    result = least_squares(resid_var, x0_var, bounds=(low_var, high_var),
                                           method="trf", max_nfev=200, ftol=1e-8, xtol=1e-8)

                    # Reconstruct full params
                    x_ref = list(x0)
                    j = 0
                    for i in range(len(varying)):
                        if varying[i]: x_ref[i] = result.x[j]; j += 1

                    y_ref = riet_calc_pattern(x_ref, np.array(tt_raw), hkl_refs_r, bg_ord,
                                              raff_a, raff_b, raff_c, a, b, c,
                                              wavelength, alpha, beta, gamma)
                    res_ref = np.array(intens_raw) - y_ref
                    chi2_ref = np.sum(res_ref**2 / (np.array(intens_raw) + 1))

                    st.session_state["riet_y_ref"] = y_ref
                    st.session_state["riet_x_ref"] = x_ref
                    st.session_state["riet_result"] = result
                    st.session_state["riet_chi2_ref"] = chi2_ref
                    st.rerun()

    # ─── Step 4: Show refined results (if done) ───
    if st.session_state.get("riet_y_ref") is not None:
        y_ref = st.session_state["riet_y_ref"]
        x_ref = st.session_state["riet_x_ref"]
        result = st.session_state.get("riet_result")
        chi2_ref = st.session_state["riet_chi2_ref"]
        hkl_refs_r = st.session_state.get("riet_hkl_refs", [])
        x0 = st.session_state.get("riet_x0", [])
        raff_a = st.session_state.get("riet_ref_a", False)
        raff_b = st.session_state.get("riet_ref_b", False)
        raff_c = st.session_state.get("riet_ref_c", False)
        bg_ord = st.session_state.get("riet_bg_order", 3)

        y_obs = np.array(intens_raw)
        res_ref = y_obs - y_ref
        s_res = np.sum(res_ref**2); s_obs = np.sum(y_obs**2)
        rp_ref = np.sum(np.abs(res_ref)) / np.sum(np.abs(y_obs)) * 100
        rwp_ref = math.sqrt(s_res / s_obs) * 100

        st.success(f"✅ Verfeinerung konvergiert ({result.nfev} Iterationen, {result.status})")

        # ─── Refined Parameters ───
        st.markdown("#### 📋 Verfeinerte Parameter")
        pnames = ["Skalenfaktor", "Zero-Shift (°)", "Caglioti U", "Caglioti V", "Caglioti W"]
        pdata = []
        for i, name in enumerate(pnames):
            pdata.append({"Parameter": name, "Initial": f"{x0[i]:.6f}", "Verfeinert": f"{x_ref[i]:.6f}"})
        idx = 5
        if raff_a: pdata.append({"Parameter": "a (Å)", "Initial": f"{a:.4f}", "Verfeinert": f"{x_ref[idx]:.4f}"}); idx += 1
        if raff_b: pdata.append({"Parameter": "b (Å)", "Initial": f"{b:.4f}", "Verfeinert": f"{x_ref[idx]:.4f}"}); idx += 1
        if raff_c: pdata.append({"Parameter": "c (Å)", "Initial": f"{c:.4f}", "Verfeinert": f"{x_ref[idx]:.4f}"}); idx += 1
        st.dataframe(pdata, use_container_width=True, hide_index=True)

        # ─── Final Pattern ───
        st.markdown("### 📈 Finales Pattern")
        fig_rf, (ax_rf, ax_df) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                                               gridspec_kw={"height_ratios": [3, 1]})
        ax_rf.plot(tt_raw, intens_raw, color="black", lw=0.8, label="I_obs")
        ax_rf.plot(tt_raw, y_ref, color="red", lw=0.8, label="I_calc")
        for r in hkl_refs_r:
            ax_rf.axvline(r["2θ (°)"], color="green", lw=0.3, alpha=0.3, ls="--")
        ax_rf.set_ylabel("Intensität"); ax_rf.legend(fontsize=9); ax_rf.grid(True, alpha=0.3)
        ax_rf.set_title(f"Rietveld-Plot (verfeinert) — χ² = {chi2_ref:.1f}", fontsize=11)
        ax_df.plot(tt_raw, res_ref, color="gray", lw=0.6)
        ax_df.axhline(0, color="black", lw=0.5)
        ax_df.fill_between(tt_raw, res_ref, 0, alpha=0.3, color="gray")
        ax_df.set_xlabel("2θ (°)"); ax_df.set_ylabel("I_obs − I_calc"); ax_df.grid(True, alpha=0.3)
        fig_rf.tight_layout(); st.pyplot(fig_rf)

        col_rc1, col_rc2, col_rc3, col_rc4 = st.columns(4)
        col_rc1.metric("Rₚ (%)", f"{rp_ref:.2f}")
        col_rc2.metric("R_wₚ (%)", f"{rwp_ref:.2f}")
        col_rc3.metric("χ²", f"{chi2_ref:.1f}")
        col_rc4.metric("Red. χ²", f"{chi2_ref/(len(tt_raw)-len(result.x)):.2f}" if len(tt_raw)>len(result.x) else "—")

st.caption("FellX4 — mit HKL-Suche & Strukturfaktoren 🚀🔷")
