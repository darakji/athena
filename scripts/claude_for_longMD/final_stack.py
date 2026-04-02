"""
Li–LLZO Interface Stacker — Long MD Structures
===============================================

Reads already lattice-matched pairs from MATCHED_DIR.
Each combo dir has exactly two CIFs — one Li slab, one LLZO slab —
with identical a,b and different c (which includes vacuum).

Steps:
1. Strip vacuum from both slabs.
2. Stack: | 7.5 Å | LLZO | 2 Å gap | Li | 7.5 Å |
3. Skip any combo containing 'Li_111'.
4. Save to OUT_DIR with naming: Li_XXX_slab__longMDstack__LLZO_YYY.cif

Author: Mehul Darak et al.
"""

import os
import numpy as np
from ase.io import read, write
from ase import Atoms

# =============================================================================
# Configuration
# =============================================================================

MATCHED_DIR = "/home/mehuldarak/athena/li_and_llzo_unrelaxed_seperate"
OUT_DIR     = "/home/mehuldarak/athena/li_and_llzo_stacked_for_long_MD_claude"

GAP        = 2.0   # Å — interface gap
VACUUM_PAD = 7.5   # Å — vacuum on each outer face

LI_TAG   = "__scaled_to_"
LLZO_TAG = "__fixed.cif"

SKIP = ["li_111"]   # case-insensitive

# =============================================================================
# Vacuum stripping
# =============================================================================

def strip_vacuum(atoms):
    atoms = atoms.copy()
    cell  = atoms.cell.array.copy()
    c_len = np.linalg.norm(cell[2])
    c_hat = cell[2] / c_len
    frac_z = np.array([np.dot(pos, c_hat) / c_len for pos in atoms.positions])
    z_min, z_max = frac_z.min(), frac_z.max()
    atoms.positions -= (z_min - 0.001) * cell[2]
    new_c_frac = (z_max - z_min) + 0.003
    new_cell = cell.copy()
    new_cell[2] = c_hat * (new_c_frac * c_len)
    atoms.set_cell(new_cell, scale_atoms=False)
    atoms.wrap()
    return atoms

# =============================================================================
# Interface builder
# =============================================================================

def build_interface(llzo, li, gap=GAP, vac=VACUUM_PAD):
    cell_llzo = llzo.cell.array
    c_llzo    = np.linalg.norm(cell_llzo[2])
    c_li      = np.linalg.norm(li.cell.array[2])
    c_hat     = cell_llzo[2] / c_llzo

    c_total   = vac + c_llzo + gap + c_li + vac

    new_cell      = np.zeros((3, 3))
    new_cell[0]   = cell_llzo[0]
    new_cell[1]   = cell_llzo[1]
    new_cell[2]   = c_hat * c_total

    llzo_pos = llzo.positions.copy() + vac * c_hat
    li_pos   = li.positions.copy()   + (vac + c_llzo + gap) * c_hat

    return Atoms(
        symbols=list(llzo.get_chemical_symbols()) + list(li.get_chemical_symbols()),
        positions=np.vstack([llzo_pos, li_pos]),
        cell=new_cell,
        pbc=True,
    )

# =============================================================================
# File discovery
# =============================================================================

def find_pair(combo_dir):
    cifs = [f for f in os.listdir(combo_dir) if f.endswith(".cif")]
    li_path = llzo_path = None
    for f in cifs:
        if LI_TAG in f:
            li_path = os.path.join(combo_dir, f)
        if f.endswith(LLZO_TAG):
            llzo_path = os.path.join(combo_dir, f)
    if li_path is None or llzo_path is None:
        raise ValueError(f"Could not find pair in {combo_dir}. Found: {cifs}")
    return li_path, llzo_path

# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    combo_dirs = sorted([
        d for d in os.listdir(MATCHED_DIR)
        if os.path.isdir(os.path.join(MATCHED_DIR, d))
    ])

    print(f"Found {len(combo_dirs)} combo(s) in {MATCHED_DIR}\n")

    for combo_name in combo_dirs:
        if any(s in combo_name.lower() for s in SKIP):
            print(f"[SKIP] {combo_name}")
            continue

        combo_path = os.path.join(MATCHED_DIR, combo_name)
        try:
            li_path, llzo_path = find_pair(combo_path)
        except ValueError as e:
            print(f"[SKIP] {e}")
            continue

        li_raw   = read(li_path)
        llzo_raw = read(llzo_path)

        li   = strip_vacuum(li_raw)
        llzo = strip_vacuum(llzo_raw)

        c_li   = np.linalg.norm(li.cell.array[2])
        c_llzo = np.linalg.norm(llzo.cell.array[2])

        # Sanity: a,b should already match
        a_li   = np.linalg.norm(li.cell.array[0])
        b_li   = np.linalg.norm(li.cell.array[1])
        a_llzo = np.linalg.norm(llzo.cell.array[0])
        b_llzo = np.linalg.norm(llzo.cell.array[1])
        da, db = abs(a_li - a_llzo), abs(b_li - b_llzo)
        if da > 0.05 or db > 0.05:
            print(f"[WARN] {combo_name}: a/b mismatch Δa={da:.4f} Δb={db:.4f} Å")

        interface = build_interface(llzo, li)
        n_total   = len(interface)
        c_total   = np.linalg.norm(interface.cell.array[2])

        # Output name: Li_XXX_slab__longMDstack__LLZO_YYY.cif
        # Extract Li surface tag (e.g. Li_100_slab) and LLZO tag
        li_tag   = combo_name.split("__")[0]          # e.g. Li_100_slab
        llzo_tag = combo_name.split("__", 1)[1]       # e.g. LLZO_001_Zr_code93_sto
        out_name = f"{li_tag}__longMDstack__{llzo_tag}.cif"
        out_path = os.path.join(OUT_DIR, out_name)

        write(out_path, interface)

        print(f"[OK] {out_name}")
        print(f"     LLZO: {len(llzo)} atoms, c={c_llzo:.3f} Å")
        print(f"     Li  : {len(li)} atoms,   c={c_li:.3f} Å")
        print(f"     Total: {n_total} atoms,  c_total={c_total:.3f} Å")
        print(f"     a={a_llzo:.4f} Å  b={b_llzo:.4f} Å\n")

    print(f"✓ Done. Structures saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()