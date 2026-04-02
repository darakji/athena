import os
import numpy as np
from ase.io import read, write
from mace.calculators import MACECalculator

# ==================================================
# Base paths
# ==================================================
IN_BASE = "/home/mehuldarak/athena/li_and_llzo_final_interfaces_2500_3000"
OUT_BASE = "/home/mehuldarak/athena/li_llzo_unrelaxed_stacking_2500_3000"
LOG_FILE = "/home/mehuldarak/athena/li_llzo_gap_scan_all_2500_3000.md"

os.makedirs(OUT_BASE, exist_ok=True)

# ==================================================
# MACE calculator
# ==================================================
calc = MACECalculator(
    model_paths=[
       "/home/mehuldarak/MACE_models/universal_09072025/mace-omat-0-medium.model"
    ],
    device="cuda",      # change to cpu if needed
    default_dtype="float32",
)

# ==================================================
# Gap scan setup
# ==================================================
TARGET_GAPS = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]

# ==================================================
# Markdown log header
# ==================================================
with open(LOG_FILE, "w") as f:
    f.write("# Li–LLZO Interface Gap Scan (All Interfaces)\n\n")
    f.write(
        "| Interface | Gap (Å) | Energy (eV) | d_min (Å) | "
        "Atom i | Elem i | Atom j | Elem j |\n"
    )
    f.write(
        "|-----------|---------|-------------|-----------|"
        "--------|--------|--------|--------|\n"
    )

# ==================================================
# Loop over all interface directories
# ==================================================
for interface_dir in sorted(os.listdir(IN_BASE)):

    interface_path = os.path.join(IN_BASE, interface_dir)
    if not os.path.isdir(interface_path):
        continue

    print(f"\nProcessing: {interface_dir}")

    # ----------------------------------------------
    # Identify CIFs
    # ----------------------------------------------
    li_cif = None
    llzo_cif = None

    for f in os.listdir(interface_path):
        if not f.endswith(".cif"):
            continue
        if "scaled_to_LLZO" in f:
            li_cif = os.path.join(interface_path, f)
        elif f.endswith("__fixed.cif"):
            llzo_cif = os.path.join(interface_path, f)

    if li_cif is None or llzo_cif is None:
        print(f"  ⚠️ Skipping (missing CIFs)")
        continue

    li = read(li_cif)
    llzo = read(llzo_cif)

    # Reference z
    z_llzo_max = llzo.positions[:, 2].max()
    z_li_min = li.positions[:, 2].min()

    # Output directory (mirror structure)
    out_dir = os.path.join(OUT_BASE, interface_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------------
    # Gap scan
    # ----------------------------------------------
    for gap in TARGET_GAPS:

        # Correct shift
        dz = gap + z_llzo_max - z_li_min
        li_shifted = li.copy()
        li_shifted.positions[:, 2] += dz

        # Stack
        stacked = llzo + li_shifted
        stacked.set_cell(llzo.cell.copy())
        stacked.set_pbc(llzo.pbc)

        # Vacuum handling
        z_all = stacked.positions[:, 2]
        z_span = z_all.max() - z_all.min()

        cell = stacked.cell.array.copy()
        cell[2, 2] = z_span + 20.0
        stacked.set_cell(cell, scale_atoms=False)
        stacked.center(axis=2)

        # Energy
        stacked.calc = calc
        energy = stacked.get_potential_energy()

        # Global min distance + pair
        D = stacked.get_all_distances(mic=False)
        np.fill_diagonal(D, np.inf)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        d_min = D[i, j]

        sym_i = stacked[i].symbol
        sym_j = stacked[j].symbol

        # Write structure
        out_cif = os.path.join(out_dir, f"Li_LLZO_gap_{gap:.2f}A.cif")
        write(out_cif, stacked)

        # Log
        with open(LOG_FILE, "a") as f:
            f.write(
                f"| {interface_dir} | {gap:.2f} | {energy:.6f} | {d_min:.3f} | "
                f"{i} | {sym_i} | {j} | {sym_j} |\n"
            )

        print(
            f"  Gap {gap:.2f} Å | "
            f"E = {energy:.4f} eV | "
            f"d_min = {d_min:.3f} Å | "
            f"pair = {sym_i}-{sym_j}"
        )

print("\n✔ All interfaces processed")
print(f"✔ Log written to {LOG_FILE}")
print(f"✔ Structures written under {OUT_BASE}")