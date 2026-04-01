import os
import numpy as np
from itertools import product
from ase.io import read, write
from ase.build import make_supercell

# ==================================================
# Paths
# ==================================================
li_dir = "/home/mehuldarak/athena/li_slabs"
llzo_dir = "/home/mehuldarak/athena/llzo_slabs"

out_base = "/home/mehuldarak/athena/li_and_llzo_final_interfaces_2500_3000"
log_file = "/home/mehuldarak/athena/li_llzo_lattice_matching_report_2500_3000.md"

os.makedirs(out_base, exist_ok=True)

# ==================================================
# Parameters
# ==================================================
MAX_REPEAT = 5
MAX_MISMATCH = 0.1  # 5%

TARGET_MIN_ATOMS = 2000
TARGET_MAX_ATOMS = 3000

# ==================================================
# Collect CIF files
# ==================================================
li_files = sorted(f for f in os.listdir(li_dir) if f.endswith(".cif"))
llzo_files = sorted(f for f in os.listdir(llzo_dir) if f.endswith(".cif"))

# ==================================================
# Markdown log header
# ==================================================
with open(log_file, "w") as f:
    f.write("# Li–LLZO Final Interface Report\n\n")
    f.write(
        "| Li slab | LLZO slab | Li supercell | LLZO supercell | "
        "Z repeat | Total atoms | a mismatch (%) | b mismatch (%) | "
        "Li scale a | Li scale b |\n"
    )
    f.write(
        "|--------|-----------|--------------|----------------|-----------|"
        "-------------|---------------|---------------|-----------|-----------|\n"
    )

# ==================================================
# Main loop
# ==================================================
for li_name, llzo_name in product(li_files, llzo_files):

    li = read(os.path.join(li_dir, li_name))
    llzo = read(os.path.join(llzo_dir, llzo_name))

    best = None

    # --------------------------------------------------
    # STEP 1: Find best lattice match (1–5, <5%)
    # --------------------------------------------------
    for li_i, li_j, llzo_i, llzo_j in product(
        range(1, MAX_REPEAT + 1),
        range(1, MAX_REPEAT + 1),
        range(1, MAX_REPEAT + 1),
        range(1, MAX_REPEAT + 1),
    ):
        li_sc_tmp = make_supercell(li, np.diag([li_i, li_j, 1]))
        llzo_sc_tmp = make_supercell(llzo, np.diag([llzo_i, llzo_j, 1]))

        cell_li = li_sc_tmp.cell.array
        cell_llzo = llzo_sc_tmp.cell.array

        a_li, b_li = np.linalg.norm(cell_li[0]), np.linalg.norm(cell_li[1])
        a_llzo, b_llzo = np.linalg.norm(cell_llzo[0]), np.linalg.norm(cell_llzo[1])

        mismatch_a = abs(a_li - a_llzo) / max(a_li, a_llzo)
        mismatch_b = abs(b_li - b_llzo) / max(b_li, b_llzo)

        # physics filter
        if mismatch_a > MAX_MISMATCH or mismatch_b > MAX_MISMATCH:
            continue

        score = max(mismatch_a, mismatch_b)

        if best is None or score < best[0]:
            best = (
                score,
                (li_i, li_j, 1),
                (llzo_i, llzo_j, 1),
                a_li,
                b_li,
                a_llzo,
                b_llzo,
                mismatch_a,
                mismatch_b
            )

    if best is None:
        print(f"[SKIPPED] No good match for {li_name} + {llzo_name}")
        continue

    _, li_rep, llzo_rep, a_li, b_li, a_llzo, b_llzo, mismatch_a, mismatch_b = best

    # --------------------------------------------------
    # STEP 2: Build matched interface
    # --------------------------------------------------
    li_sc = make_supercell(li, np.diag(li_rep))
    llzo_sc = make_supercell(llzo, np.diag(llzo_rep))

    # scale Li
    cell_li = li_sc.cell.array.copy()
    scale_a = a_llzo / a_li
    scale_b = b_llzo / b_li

    cell_li[0] *= scale_a
    cell_li[1] *= scale_b

    li_sc.set_cell(cell_li, scale_atoms=True)

    # --------------------------------------------------
    # STEP 3: Auto-scale in Z to reach 2000–3000 atoms
    # --------------------------------------------------
    base_atoms = len(li_sc) + len(llzo_sc)

    z_repeat = 1
    while base_atoms * z_repeat < TARGET_MIN_ATOMS:
        z_repeat += 1

    # if overshoot too much, step back
    if base_atoms * z_repeat > TARGET_MAX_ATOMS:
        z_repeat -= 1

    z_repeat = max(1, z_repeat)

    li_sc = make_supercell(li_sc, np.diag([1, 1, z_repeat]))
    llzo_sc = make_supercell(llzo_sc, np.diag([1, 1, z_repeat]))

    total_atoms = len(li_sc) + len(llzo_sc)

    # --------------------------------------------------
    # STEP 4: Save
    # --------------------------------------------------
    li_tag = os.path.splitext(li_name)[0]
    llzo_tag = os.path.splitext(llzo_name)[0]

    li_rep_str = f"{li_rep[0]}x{li_rep[1]}x{li_rep[2]}"
    llzo_rep_str = f"{llzo_rep[0]}x{llzo_rep[1]}x{llzo_rep[2]}"

    combo_dir = os.path.join(out_base, f"{li_tag}__{llzo_tag}")
    os.makedirs(combo_dir, exist_ok=True)

    li_out = f"{li_tag}__sc_{li_rep_str}_z{z_repeat}__scaled.cif"
    llzo_out = f"{llzo_tag}__sc_{llzo_rep_str}_z{z_repeat}__fixed.cif"

    write(os.path.join(combo_dir, li_out), li_sc)
    write(os.path.join(combo_dir, llzo_out), llzo_sc)

    # --------------------------------------------------
    # STEP 5: Log
    # --------------------------------------------------
    with open(log_file, "a") as f:
        f.write(
            f"| {li_name} | {llzo_name} | {li_rep} | {llzo_rep} | "
            f"{z_repeat} | {total_atoms} | "
            f"{mismatch_a*100:.3f} | {mismatch_b*100:.3f} | "
            f"{scale_a:.5f} | {scale_b:.5f} |\n"
        )

print("Done: clean interfaces + size-controlled systems generated.")
print(f"Report: {log_file}")
print(f"Structures: {out_base}/")