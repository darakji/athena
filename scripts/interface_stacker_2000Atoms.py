"""
Li–LLZO Interface Stacker (v2)
==============================

Stacks pre-matched Li and LLZO slabs into interface structures:

    | 7.5 Å vacuum | LLZO | gap | Li | 7.5 Å vacuum |

Strategy to reach TARGET_ATOMS (≥2000)
---------------------------------------
1. Strip vacuum from both slabs.
2. Repeat Li along c ONLY until c_Li ≈ c_LLZO (capped at MAX_LI_C_REPEATS).
3. Stack the interface.
4. Expand the WHOLE interface in-plane (a×b) using the smallest (m×n)
   supercell that pushes total atoms >= TARGET_ATOMS.
   Search order: sorted by total atoms to find minimum expansion.

This keeps both slabs at comparable thickness and avoids runaway Li stacking.

Layout (along c-axis):
    [0 → 7.5 Å]              vacuum (LLZO side)
    [7.5 → 7.5 + c_llzo]     LLZO atoms
    [... + gap]               interface gap
    [... + c_li * li_repeats] Li atoms
    [end → end + 7.5 Å]      vacuum (Li side)

Dependencies
------------
- ASE
- NumPy

Author
------
Mehul Darak et al.
"""

import os
import numpy as np
from ase.io import read, write
from ase.build import make_supercell
from ase import Atoms

# =============================================================================
# Configuration
# =============================================================================

MATCHED_DIR = "/home/mehuldarak/athena/li_and_llzo_unrelaxed_seperate"
OUT_DIR     = "/home/mehuldarak/athena/li_llzo_interfaces_claude_2000"
LOG_FILE    = "/home/mehuldarak/athena/li_llzo_interface_report_claude_2000.md"

TARGET_GAPS   = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]  # Angstroms
VACUUM_PAD    = 7.5   # Angstroms on each outer face
TARGET_ATOMS  = 2000

# Li is repeated along c until c_Li >= c_LLZO * this fraction, then stops.
# 0.8 means: stop repeating Li when it's at least 80% as thick as LLZO.
LI_C_THICKNESS_RATIO = 0.8
MAX_LI_C_REPEATS = 4   # hard cap — never repeat Li more than this along c

# In-plane search: try (m x n) up to this value on each axis
MAX_INPLANE_REPEAT = 6

# File tags from lattice matching step
LI_TAG   = "__scaled_to_"
LLZO_TAG = "__fixed.cif"

# =============================================================================
# Vacuum stripping
# =============================================================================

def strip_vacuum(atoms):
    """
    Remove vacuum from a slab along c.
    Shifts atoms so the lowest sits just above z=0, trims cell c to atomic extent.
    """
    atoms = atoms.copy()
    cell  = atoms.cell.array.copy()
    c_len = np.linalg.norm(cell[2])
    c_hat = cell[2] / c_len

    frac_z = np.array([np.dot(pos, c_hat) / c_len for pos in atoms.positions])

    z_min = frac_z.min()
    z_max = frac_z.max()

    # Shift so z_min -> 0.001 (small buffer at bottom)
    atoms.positions -= (z_min - 0.001) * cell[2]

    # New c = atomic extent + small buffers
    new_c_frac = (z_max - z_min) + 0.003
    new_cell = cell.copy()
    new_cell[2] = c_hat * (new_c_frac * c_len)
    atoms.set_cell(new_cell, scale_atoms=False)
    atoms.wrap()

    return atoms

# =============================================================================
# Li c-repeat: match thickness to LLZO, don't overshoot
# =============================================================================

def repeat_li_c(li_stripped, c_llzo):
    """
    Repeat Li along c until c_Li >= c_LLZO * LI_C_THICKNESS_RATIO.
    Hard-capped at MAX_LI_C_REPEATS.

    Returns (repeated_li, n_repeats).
    """
    c_li = np.linalg.norm(li_stripped.cell.array[2])
    target_c = c_llzo * LI_C_THICKNESS_RATIO

    n = 1
    while (c_li * n < target_c) and (n < MAX_LI_C_REPEATS):
        n += 1

    if n == 1:
        return li_stripped.copy(), 1

    li_rep = make_supercell(li_stripped, np.diag([1, 1, n]))
    return li_rep, n

# =============================================================================
# Build interface (single unit, before in-plane expansion)
# =============================================================================

def build_interface(llzo_stripped, li_c_repeated, gap_ang, vacuum_pad=VACUUM_PAD):
    """
    Stack:  vacuum | LLZO | gap | Li | vacuum
    All positions in Cartesian space along c_hat.
    """
    cell_llzo = llzo_stripped.cell.array
    cell_li   = li_c_repeated.cell.array

    c_llzo = np.linalg.norm(cell_llzo[2])
    c_li   = np.linalg.norm(cell_li[2])
    c_hat  = cell_llzo[2] / c_llzo

    c_total = vacuum_pad + c_llzo + gap_ang + c_li + vacuum_pad

    new_cell = np.zeros((3, 3))
    new_cell[0] = cell_llzo[0]
    new_cell[1] = cell_llzo[1]
    new_cell[2] = c_hat * c_total

    llzo_pos = llzo_stripped.positions.copy() + vacuum_pad * c_hat
    li_pos   = li_c_repeated.positions.copy() + (vacuum_pad + c_llzo + gap_ang) * c_hat

    interface = Atoms(
        symbols=(
            list(llzo_stripped.get_chemical_symbols()) +
            list(li_c_repeated.get_chemical_symbols())
        ),
        positions=np.vstack([llzo_pos, li_pos]),
        cell=new_cell,
        pbc=True,
    )
    return interface

# =============================================================================
# In-plane expansion to reach TARGET_ATOMS
# =============================================================================

def expand_inplane(interface, target=TARGET_ATOMS):
    """
    Find the smallest (m x n x 1) supercell of `interface` with
    total atoms >= target. Searches m, n in 1..MAX_INPLANE_REPEAT,
    sorted by atom count to find minimum expansion.

    Returns (expanded_atoms, m, n).
    """
    n_base = len(interface)

    if n_base >= target:
        return interface.copy(), 1, 1

    # Generate all (m, n) candidates sorted by total atoms
    candidates = []
    for m in range(1, MAX_INPLANE_REPEAT + 1):
        for n in range(1, MAX_INPLANE_REPEAT + 1):
            total = n_base * m * n
            candidates.append((total, m, n))

    candidates.sort(key=lambda x: x[0])

    for total, m, n in candidates:
        if total >= target:
            expanded = make_supercell(interface, np.diag([m, n, 1]))
            return expanded, m, n

    # Fallback: use largest searched
    total, m, n = candidates[-1]
    expanded = make_supercell(interface, np.diag([m, n, 1]))
    print(f"  [WARNING] Could not reach {target} atoms. "
          f"Best: {total} atoms with ({m}x{n}x1). "
          f"Increase MAX_INPLANE_REPEAT if needed.")
    return expanded, m, n

# =============================================================================
# File discovery
# =============================================================================

def find_li_llzo_files(combo_dir):
    cifs = [f for f in os.listdir(combo_dir) if f.endswith(".cif")]
    li_path = llzo_path = None
    for f in cifs:
        if LI_TAG in f:
            li_path = os.path.join(combo_dir, f)
        if f.endswith(LLZO_TAG):
            llzo_path = os.path.join(combo_dir, f)
    if li_path is None or llzo_path is None:
        raise ValueError(
            f"Could not identify Li+LLZO pair in {combo_dir}.\n"
            f"Found: {cifs}"
        )
    return li_path, llzo_path

# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(LOG_FILE, "w") as f:
        f.write("# Li–LLZO Interface Stacking Report (v2)\n\n")
        f.write(
            "| Pair | Gap (Å) | Li c-reps | In-plane (m×n) | "
            "N(LLZO) | N(Li) | N(total) | "
            "c_LLZO (Å) | c_Li (Å) | c_total (Å) | Output |\n"
        )
        f.write(
            "|------|---------|-----------|----------------|"
            "---------|-------|----------|"
            "-----------|---------|-------------|--------|\n"
        )

    combo_dirs = sorted([
        d for d in os.listdir(MATCHED_DIR)
        if os.path.isdir(os.path.join(MATCHED_DIR, d))
    ])

    if not combo_dirs:
        print(f"No subdirectories found in {MATCHED_DIR}.")
        return

    print(f"Found {len(combo_dirs)} matched pair(s).\n")

    for combo_name in combo_dirs:
        combo_path = os.path.join(MATCHED_DIR, combo_name)

        try:
            li_path, llzo_path = find_li_llzo_files(combo_path)
        except ValueError as e:
            print(f"  [SKIP] {combo_name}: {e}")
            continue

        print(f"Processing: {combo_name}")

        li_raw   = read(li_path)
        llzo_raw = read(llzo_path)

        li_stripped   = strip_vacuum(li_raw)
        llzo_stripped = strip_vacuum(llzo_raw)

        c_llzo    = np.linalg.norm(llzo_stripped.cell.array[2])
        c_li_base = np.linalg.norm(li_stripped.cell.array[2])

        print(f"  Vacuum stripped — LLZO: {c_llzo:.2f} Å ({len(llzo_stripped)} atoms) | "
              f"Li: {c_li_base:.2f} Å ({len(li_stripped)} atoms)")

        # Repeat Li along c to match LLZO thickness (not to hit atom count)
        li_c_rep, n_c = repeat_li_c(li_stripped, c_llzo)
        c_li = np.linalg.norm(li_c_rep.cell.array[2])
        print(f"  Li c-repeated x{n_c} → {c_li:.2f} Å, {len(li_c_rep)} atoms")

        pair_out = os.path.join(OUT_DIR, combo_name)
        os.makedirs(pair_out, exist_ok=True)

        for gap in TARGET_GAPS:
            # Build unit interface
            interface_unit = build_interface(llzo_stripped, li_c_rep, gap_ang=gap)

            # Expand in-plane to reach target atoms
            interface_final, m, n = expand_inplane(interface_unit)

            n_total  = len(interface_final)
            c_total  = np.linalg.norm(interface_final.cell.array[2])
            n_llzo_f = len(llzo_stripped) * m * n
            n_li_f   = len(li_c_rep) * m * n

            out_name = (
                f"{combo_name}"
                f"__gap_{gap:.1f}A"
                f"__Lic{n_c}_ip{m}x{n}"
                f"__N{n_total}.cif"
            )
            write(os.path.join(pair_out, out_name), interface_final)

            print(f"  gap={gap:.1f} Å | in-plane {m}×{n} | N={n_total} | "
                  f"c={c_total:.2f} Å → {out_name}")

            with open(LOG_FILE, "a") as f:
                f.write(
                    f"| {combo_name} | {gap:.1f} | {n_c} | {m}×{n} | "
                    f"{n_llzo_f} | {n_li_f} | {n_total} | "
                    f"{c_llzo:.3f} | {c_li:.3f} | {c_total:.3f} | {out_name} |\n"
                )

        print()

    print("✓ Done.")
    print(f"  Structures : {OUT_DIR}/")
    print(f"  Report     : {LOG_FILE}")


if __name__ == "__main__":
    main()