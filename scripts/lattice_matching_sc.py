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

out_base = "li_and_llzo_unrelaxed_seperate"
log_file = "li_llzo_lattice_matching_report.md"

os.makedirs(out_base, exist_ok=True)

# ==================================================
# Parameters
# ==================================================
MAX_REPEAT = 5  # search (1..5)x(1..5)x1 supercells

# ==================================================
# Collect CIF files
# ==================================================
li_files = sorted(f for f in os.listdir(li_dir) if f.endswith(".cif"))
llzo_files = sorted(f for f in os.listdir(llzo_dir) if f.endswith(".cif"))

# ==================================================
# Markdown log header
# ==================================================
with open(log_file, "w") as f:
    f.write("# Li–LLZO Lattice Matching Report\n\n")
    f.write(
        "| Li slab | LLZO slab | Li supercell | LLZO supercell | "
        "a mismatch (%) | b mismatch (%) | "
        "Li scale a | Li scale b |\n"
    )
    f.write(
        "|--------|-----------|--------------|----------------|"
        "---------------|---------------|-----------|-----------|\n"
    )

# ==================================================
# Main loop over all combinations
# ==================================================
for li_name, llzo_name in product(li_files, llzo_files):

    li = read(os.path.join(li_dir, li_name))
    llzo = read(os.path.join(llzo_dir, llzo_name))

    best = None
    # best = (score, li_rep, llzo_rep, a_li, b_li, a_llzo, b_llzo)

    # --------------------------------------------------
    # Supercell search
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
            )

    # --------------------------------------------------
    # Extract best match
    # --------------------------------------------------
    _, li_rep, llzo_rep, a_li, b_li, a_llzo, b_llzo = best

    # --------------------------------------------------
    # Build final supercells
    # --------------------------------------------------
    li_sc = make_supercell(li, np.diag(li_rep))
    llzo_sc = make_supercell(llzo, np.diag(llzo_rep))

    cell_li = li_sc.cell.array.copy()

    # --------------------------------------------------
    # SCALE ONLY Li (axis-wise)
    # --------------------------------------------------
    scale_a = a_llzo / a_li
    scale_b = b_llzo / b_li

    cell_li[0] *= scale_a
    cell_li[1] *= scale_b

    li_sc.set_cell(cell_li, scale_atoms=True)

    # --------------------------------------------------
    # Unique naming
    # --------------------------------------------------
    li_tag = os.path.splitext(li_name)[0]
    llzo_tag = os.path.splitext(llzo_name)[0]

    li_rep_str = f"{li_rep[0]}x{li_rep[1]}x{li_rep[2]}"
    llzo_rep_str = f"{llzo_rep[0]}x{llzo_rep[1]}x{llzo_rep[2]}"

    combo_dir = os.path.join(
        out_base,
        f"{li_tag}__{llzo_tag}"
    )
    os.makedirs(combo_dir, exist_ok=True)

    li_out = (
        f"{li_tag}__sc_{li_rep_str}"
        f"__scaled_to_{llzo_tag}.cif"
    )
    llzo_out = (
        f"{llzo_tag}__sc_{llzo_rep_str}"
        f"__fixed.cif"
    )

    write(os.path.join(combo_dir, li_out), li_sc)
    write(os.path.join(combo_dir, llzo_out), llzo_sc)

    # --------------------------------------------------
    # Log results
    # --------------------------------------------------
    with open(log_file, "a") as f:
        f.write(
            f"| {li_name} | {llzo_name} | {li_rep} | {llzo_rep} | "
            f"{abs(a_li - a_llzo)/max(a_li,a_llzo)*100:.3f} | "
            f"{abs(b_li - b_llzo)/max(b_li,b_llzo)*100:.3f} | "
            f"{scale_a:.5f} | {scale_b:.5f} |\n"
        )

print("All Li–LLZO combinations processed successfully.")
print(f"Results written to: {log_file}")
print(f"Structures written under: {out_base}/")