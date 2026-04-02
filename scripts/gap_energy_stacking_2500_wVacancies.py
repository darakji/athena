import os
import numpy as np
import shutil
from ase.io import read, write
from mace.calculators import MACECalculator

# ==================================================
# Paths
# ==================================================
IN_BASE = "/home/mehuldarak/athena/li_and_llzo_final_interfaces_2500_3000"
OUT_BASE = "/home/mehuldarak/athena/li_llzo_unrelaxed_stacking_2500_3000"
LOG_FILE = os.path.join(OUT_BASE, "li_llzo_gap_scan_all_2500_3000.md")

FINAL_DIR = os.path.join(OUT_BASE, "final_selected")

os.makedirs(OUT_BASE, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# ==================================================
# MACE
# ==================================================
calc = MACECalculator(
    model_paths=[
        "/home/mehuldarak/athena/mace_fps_training/checkpoints/mace_fps_split17_SaveIT0_256_candidate3.model"
    ],
    device="cuda",
    default_dtype="float32",
    batch_size=16
)

# ==================================================
# Gap scan
# ==================================================
TARGET_GAPS = np.linspace(4.0, 1.0, 13)

EV_TO_J = 1.60218e-19
ANG2_TO_M2 = 1e-20

# ==================================================
# Log header
# ==================================================
with open(LOG_FILE, "w") as f:
    f.write("# Li–LLZO Interface Gap Scan (Vacancy Structures)\n\n")
    f.write(
        "| Interface | Gap | E_stack | E_Li | E_LLZO | "
        "E_int | Area | γ (J/m²) | d_min | pair | BEST |\n"
    )
    f.write(
        "|-----------|-----|---------|------|--------|"
        "--------|--------|-----------|--------|------|------|\n"
    )

# ==================================================
# Main loop
# ==================================================
for interface_dir in sorted(os.listdir(IN_BASE)):

    path = os.path.join(IN_BASE, interface_dir)
    if not os.path.isdir(path):
        continue

    print(f"\nProcessing: {interface_dir}")

    li_cif, llzo_cif = None, None

    for f in os.listdir(path):
        if f.endswith("__scaled.cif"):
            li_cif = os.path.join(path, f)
        elif f.endswith("__vacancy.cif"):
            llzo_cif = os.path.join(path, f)

    if li_cif is None or llzo_cif is None:
        print("  ⚠️ Missing files")
        continue

    li = read(li_cif)
    llzo = read(llzo_cif)

    # --------------------------------------------------
    # Bulk energies
    # --------------------------------------------------
    li.calc = calc
    llzo.calc = calc

    E_li = li.get_potential_energy()
    E_llzo = llzo.get_potential_energy()

    # --------------------------------------------------
    # Area
    # --------------------------------------------------
    cell = llzo.cell.array
    area = np.linalg.norm(np.cross(cell[0], cell[1]))

    results = []

    z_llzo_max = llzo.positions[:, 2].max()
    z_li_min = li.positions[:, 2].min()

    out_dir = os.path.join(OUT_BASE, interface_dir)
    os.makedirs(out_dir, exist_ok=True)

    for gap in TARGET_GAPS:

        dz = gap + z_llzo_max - z_li_min

        li_shifted = li.copy()
        li_shifted.positions[:, 2] += dz

        stacked = llzo + li_shifted
        stacked.set_cell(llzo.cell.copy())
        stacked.set_pbc(llzo.pbc)

        # vacuum
        z = stacked.positions[:, 2]
        span = z.max() - z.min()

        cell_new = stacked.cell.array.copy()
        cell_new[2, 2] = span + 20
        stacked.set_cell(cell_new, scale_atoms=False)
        stacked.center(axis=2)

        # --------------------------------------------------
        # Energy
        # --------------------------------------------------
        stacked.calc = calc
        E_stack = stacked.get_potential_energy()

        E_int = E_stack - (E_li + E_llzo)

        # 🔥 Correct formula (divide by A, NOT 2A)
        gamma = (E_int * EV_TO_J) / (2*area * ANG2_TO_M2)

        # --------------------------------------------------
        # Interface-only distance
        # --------------------------------------------------
        n_llzo = len(llzo)
        n_total = len(stacked)

        D = stacked.get_all_distances(mic=False)

        d_min = np.inf
        pair = ("", "")

        for i in range(n_llzo):
            for j in range(n_llzo, n_total):
                if D[i, j] < d_min:
                    d_min = D[i, j]
                    pair = (stacked[i].symbol, stacked[j].symbol)

        # save structure
        out_cif = os.path.join(out_dir, f"gap_{gap:.2f}.cif")
        write(out_cif, stacked)

        results.append((gap, E_stack, E_int, gamma, d_min, pair))

        print(f"  Gap {gap:.2f} Å | γ = {gamma:.3f} J/m²")

    # --------------------------------------------------
    # BEST GAP
    # --------------------------------------------------
    best_idx = np.argmin([r[2] for r in results])
    best_gap = results[best_idx][0]

    # --------------------------------------------------
    # SAVE BEST STRUCTURE
    # --------------------------------------------------
    src_file = os.path.join(out_dir, f"gap_{best_gap:.2f}.cif")

    dst_file = os.path.join(
        FINAL_DIR,
        f"{interface_dir}__gap_{best_gap:.2f}.cif"
    )

    if os.path.exists(src_file):
        shutil.copy(src_file, dst_file)
        print(f"  ✔ Selected best → {dst_file}")
    else:
        print(f"  ⚠️ Missing best file: {src_file}")

    # --------------------------------------------------
    # LOG
    # --------------------------------------------------
    with open(LOG_FILE, "a") as f:
        for i, (gap, E_stack, E_int, gamma, d_min, pair) in enumerate(results):

            best_flag = "⭐" if i == best_idx else ""

            f.write(
                f"| {interface_dir} | {gap:.2f} | {E_stack:.4f} | "
                f"{E_li:.4f} | {E_llzo:.4f} | {E_int:.4f} | "
                f"{area:.2f} | {gamma:.3f} | {d_min:.3f} | "
                f"{pair[0]}-{pair[1]} | {best_flag} |\n"
            )

print("\n✔ Done: All interfaces processed + best structures selected")
print(f"✔ Final structures: {FINAL_DIR}")
print(f"✔ Log file: {LOG_FILE}")