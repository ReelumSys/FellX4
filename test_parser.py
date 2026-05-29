import re, math

def parse_cif(content):
    cif = {}
    clean = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        clean.append(line)
    text = "\n".join(clean)

    for key in ["_cell_length_a", "_cell_length_b", "_cell_length_c",
                "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"]:
        m = re.search(rf"{re.escape(key)}\s+([\d.]+(?:\(\d+\))?)", text)
        if m:
            cif[key] = float(m.group(1).split("(")[0])

    m = re.search(r"_symmetry_space_group_name_H-M\s+'([^']+)'", text)
    if m:
        cif["space_group"] = m.group(1)

    lines = text.split("\n")
    in_loop = False
    headers_raw = []
    values = []
    atom_site_prefix = None

    for line in lines:
        ls = line.strip()
        if ls == "loop_":
            in_loop = True
            headers_raw = []
            values = []
            atom_site_prefix = None
            continue
        if in_loop:
            if ls.startswith("_atom_site.") or ls.startswith("_atom_site_"):
                headers_raw.append(ls)
                if ls.startswith("_atom_site."):
                    atom_site_prefix = "_atom_site."
                else:
                    atom_site_prefix = "_atom_site_"
            elif headers_raw and ls:
                if ls.startswith("_"):
                    in_loop = False
                else:
                    values.append(ls.split())
            elif not headers_raw and ls.startswith("_"):
                in_loop = False

    if headers_raw and values:
        prefix = atom_site_prefix or "_atom_site."
        header_keys = [h[len(prefix):] for h in headers_raw]
        atoms = []
        for row in values:
            if len(row) >= len(header_keys):
                atom = dict(zip(header_keys, row))
                atoms.append(atom)
        cif["atoms"] = atoms

    return cif if any(k in cif for k in ["_cell_length_a", "atoms"]) else None

# Test
cif = """data_quartz
_cell_length_a    4.9134
_cell_length_b    4.9134
_cell_length_c    5.4052
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 120.0
_symmetry_space_group_name_H-M 'P 32 2 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1 0.4699 0.0000 0.0000 1.0
O1  0.4135 0.2671 0.1191 1.0
"""
r = parse_cif(cif)
print("Underscore:", "PASS" if r and "atoms" in r else "FAIL")
if r: print("  cell_a:", r.get("_cell_length_a"), "atoms:", len(r["atoms"]))

cif2 = """data_silicon
_cell_length_a 5.430
_cell_length_b 5.430
_cell_length_c 5.430
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site.type_symbol
_atom_site.fract_x
_atom_site.fract_y
_atom_site.fract_z
Si 0.0 0.0 0.0
Si 0.5 0.5 0.5
"""
r2 = parse_cif(cif2)
print("Dot:", "PASS" if r2 and "atoms" in r2 and len(r2["atoms"]) == 2 else "FAIL")

cif3 = """data_test
_cell_length_a 5.430(5)
_cell_length_b 5.430(5)
_cell_length_c 5.430(5)
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""
r3 = parse_cif(cif3)
print("ESD:", "PASS" if r3 and r3["_cell_length_a"] == 5.43 else "FAIL")

cif4 = """data_cod
_cell_length_a 5.430
_cell_length_b 5.430
_cell_length_c 5.430
_cell_angle_alpha 90.000
_cell_angle_beta 90.000
_cell_angle_gamma 90.000
_symmetry_space_group_name_H-M 'F d -3 m'
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_thermal_displace_type
_atom_site_B_iso_or_equivalent
Si 0.00000 0.00000 0.00000 1.0 Biso 0.450
"""
r4 = parse_cif(cif4)
print("COD-style:", "PASS" if r4 and "atoms" in r4 and len(r4["atoms"])==1 else "FAIL")
if r4 and "atoms" in r4:
    for a in r4["atoms"]:
        print("  Atom:", a)
