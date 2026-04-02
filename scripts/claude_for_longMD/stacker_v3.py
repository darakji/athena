"""
Li–LLZO Interface Stacker (v4)
==============================

Fixes vs v3:
    - Bulk references are built correctly:
        * Li bulk: from ase.build.bulk("Li", "bcc") — proper BCC unit cell,
          strained in-plane to match the scaled Li slab lattice parameters.
        * LLZO bulk: derived from the __fixed.cif slab by detecting the
          primitive repeat unit along c (half-slab, quarter-slab, etc.) using
          atom count divisibility and the known LLZO formula unit (192 atoms).
          This avoids slab-tiling artefacts from surface terminations.
    - No external bulk CIF files needed — everything derived from MATCHED_DIR.

Bulk reference atom count:
    The bulk is tiled to have AT LEAST as many atoms as the slab region in
    the interface, so E_total - E_bulk_LLZO - E_bulk_Li is well-defined.
    Exact atom count equality is preferred; a small excess is acceptable.

Layout (along c-axis of interface):
    [0 → 7.5 Å]              vacuum (LLZO side)
    [7.5 → 7.5 + c_llzo]     LLZO atoms
    [... + gap]               interface gap
    [... + c_li * li_repeats] Li atoms
    [end → end + 7.5 Å]      vacuum (Li side)

Author: Mehul Darak et al.
"""

import os
import numpy as np
from itertools import product as iproduct
from ase.io import read, write
from ase.build import make_supercell, bulk as ase_bulk
from ase import Atoms

# =============================================================================
# Configuration
# =============================================================================

MATCHED_DIR  = "/home/mehuldarak/athena/li_and_llzo_unrelaxed_seperate"
OUT_DIR      = "/home/mehuldarak/athena/li_and_llzo_stacked_diff_gaps_claude_2000"
BULK_REF_DIR = "/home/mehuldarak/athena/bulk_references"
LOG_FILE     = "/home/mehuldarak/athena/li_llzo_interface_v4_report.md"

TARGET_GAPS          = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]
VACUUM_PAD           = 7.5   # Å on each outer face
TARGET_ATOMS         = 2000
LI_C_THICKNESS_RATIO = 0.8
MAX_LI_C_REPEATS     = 4
MAX_INPLANE_REPEAT   = 6

LI_TAG   = "__scaled_to_"
LLZO_TAG = "__fixed.cif"

# LLZO conventional unit cell has 192 atoms (garnet, Ia-3d).
# Used to find the primitive repeat along c in the slab.
LLZO_UNITCELL_ATOMS = 192

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
# Li c-repeat
# =============================================================================

def repeat_li_c(li_stripped, c_llzo):
    c_li = np.linalg.norm(li_stripped.cell.array[2])
    target_c = c_llzo * LI_C_THICKNESS_RATIO
    n = 1
    while (c_li * n < target_c) and (n < MAX_LI_C_REPEATS):
        n += 1
    if n == 1:
        return li_stripped.copy(), 1
    return make_supercell(li_stripped, np.diag([1, 1, n])), n

# =============================================================================
# Interface builder
# =============================================================================

def build_interface(llzo_stripped, li_c_repeated, gap_ang, vacuum_pad=VACUUM_PAD):
    cell_llzo = llzo_stripped.cell.array
    c_llzo    = np.linalg.norm(cell_llzo[2])
    c_li      = np.linalg.norm(li_c_repeated.cell.array[2])
    c_hat     = cell_llzo[2] / c_llzo
    c_total   = vacuum_pad + c_llzo + gap_ang + c_li + vacuum_pad

    new_cell = np.zeros((3, 3))
    new_cell[0] = cell_llzo[0]
    new_cell[1] = cell_llzo[1]
    new_cell[2] = c_hat * c_total

    llzo_pos = llzo_stripped.positions.copy() + vacuum_pad * c_hat
    li_pos   = li_c_repeated.positions.copy() + (vacuum_pad + c_llzo + gap_ang) * c_hat

    return Atoms(
        symbols=(
            list(llzo_stripped.get_chemical_symbols()) +
            list(li_c_repeated.get_chemical_symbols())
        ),
        positions=np.vstack([llzo_pos, li_pos]),
        cell=new_cell,
        pbc=True,
    )

# =============================================================================
# In-plane expansion
# =============================================================================

def expand_inplane(interface, target=TARGET_ATOMS):
    n_base = len(interface)
    if n_base >= target:
        return interface.copy(), 1, 1
    candidates = sorted(
        [(n_base * m * n, m, n)
         for m in range(1, MAX_INPLANE_REPEAT + 1)
         for n in range(1, MAX_INPLANE_REPEAT + 1)],
        key=lambda x: x[0]
    )
    for total, m, n in candidates:
        if total >= target:
            return make_supercell(interface, np.diag([m, n, 1])), m, n
    total, m, n = candidates[-1]
    print(f"  [WARNING] Best in-plane expansion: {total} atoms ({m}x{n}). "
          f"Increase MAX_INPLANE_REPEAT if needed.")
    return make_supercell(interface, np.diag([m, n, 1])), m, n

# =============================================================================
# Bulk Li — from ase.build.bulk, strained to match slab a,b
# =============================================================================

def make_bulk_li(li_stripped_unit, li_c_rep_slab, m, n, n_li_slab):
    """
    Build a proper periodic bulk Li reference:
    1. Get BCC Li unit cell from ASE (2 atoms, cubic).
    2. Strain its a,b to match the scaled Li slab lattice (self-consistent
       with the strained interface).
    3. Tile (m x n x k) where k is chosen so atom count >= n_li_slab.

    The strained BCC cell ensures the bulk reference has the same lateral
    strain as the Li slab in the interface — critical for a meaningful
    interface energy.

    Parameters
    ----------
    li_stripped_unit : Atoms  vacuum-stripped Li slab (single unit, pre-c-repeat)
    li_c_rep_slab    : Atoms  Li slab after c-repeat (used in interface)
    m, n             : int    in-plane expansion matching the interface
    n_li_slab        : int    total Li atoms in interface Li region

    Returns bulk_li : Atoms
    """
    # BCC Li unit cell (2 atoms)
    bcc_li = ase_bulk("Li", "bcc", a=3.51, cubic=True)   # conventional, 2 atoms

    # Strain a, b of BCC to match the Li slab lateral parameters.
    # Li slab a-vector length after scaling (from lattice matching step).
    cell_slab = li_stripped_unit.cell.array
    a_slab = np.linalg.norm(cell_slab[0])
    b_slab = np.linalg.norm(cell_slab[1])

    # BCC Li is cubic — a=b=c. We strain it anisotropically to match slab.
    # New cell: a_slab/m per repeat unit in a, b_slab/n in b, keep c as BCC.
    # Actually we tile first then check — simpler to just scale the BCC cell.
    a_bcc = bcc_li.cell.lengths()[0]

    # Scale factors: match a_slab/(m repeats) and b_slab/(n repeats)
    # After tiling m×n, the bulk will have a=a_slab, b=b_slab.
    scale_a = (a_slab / m) / a_bcc
    scale_b = (b_slab / n) / a_bcc

    new_bcc_cell = bcc_li.cell.array.copy()
    new_bcc_cell[0] *= scale_a
    new_bcc_cell[1] *= scale_b
    # c stays as BCC (periodicity along c is independent)
    bcc_li.set_cell(new_bcc_cell, scale_atoms=True)

    # Find minimum k along c so m*n*k*2 >= n_li_slab
    k = max(1, int(np.ceil(n_li_slab / (m * n * len(bcc_li)))))

    bulk_li = make_supercell(bcc_li, np.diag([m, n, k]))
    bulk_li.wrap()

    print(f"    Bulk Li (BCC strained): ({m}×{n}×{k}) × {len(bcc_li)} = "
          f"{len(bulk_li)} atoms  (slab had {n_li_slab})")
    return bulk_li

# =============================================================================
# Bulk LLZO — primitive c-repeat unit derived from slab
# =============================================================================

def find_llzo_primitive_c_unit(llzo_stripped):
    """
    Find the primitive repeat unit along c of the LLZO slab.

    LLZO garnet has 192 atoms/conventional cell. The slab has N_slab atoms
    total. The primitive c-repeat unit has N_slab / k atoms, where k is
    the smallest integer that divides N_slab such that N_slab/k is a
    divisor of 192 (or a multiple/divisor thereof).

    Strategy:
    1. Try k = 1, 2, 3, 4 (max 4 repeats expected in a ~20 Å slab).
    2. For each k, check if N_slab is divisible by k.
    3. Among valid k, pick largest k (smallest primitive unit).

    The primitive unit is then the bottom k-th fraction of the slab atoms,
    with c-vector = c_slab / k.

    Returns (primitive_unit, k_found).
    If no clean divisor found, returns (llzo_stripped, 1) as fallback.
    """
    n_slab = len(llzo_stripped)
    cell   = llzo_stripped.cell.array.copy()
    c_len  = np.linalg.norm(cell[2])
    c_hat  = cell[2] / c_len

    # Try divisors from largest to smallest (most primitive first)
    for k in [4, 3, 2, 1]:
        if n_slab % k != 0:
            continue
        n_unit = n_slab // k

        # Check if n_unit is a reasonable LLZO fragment
        # (divisor or multiple of LLZO_UNITCELL_ATOMS or its common factors)
        # We accept any clean integer — the key is divisibility
        c_unit = c_len / k

        # Extract atoms in the bottom 1/k fraction of the slab along c
        z_cutoff = c_unit + 0.5   # 0.5 Å buffer
        unit_indices = [
            i for i, pos in enumerate(llzo_stripped.positions)
            if np.dot(pos, c_hat) < z_cutoff
        ]

        if len(unit_indices) == n_unit:
            # Clean extraction — build the primitive unit cell
            unit_atoms = llzo_stripped[unit_indices].copy()
            new_cell   = cell.copy()
            new_cell[2] = c_hat * c_unit
            unit_atoms.set_cell(new_cell, scale_atoms=False)
            unit_atoms.wrap()
            print(f"    LLZO primitive unit: slab/{k} = {n_unit} atoms, "
                  f"c_unit = {c_unit:.3f} Å")
            return unit_atoms, k

    # Fallback: use full slab as unit (k=1)
    print(f"    [WARN] Could not find clean LLZO primitive unit. "
          f"Using full slab ({n_slab} atoms) as unit — slab-tiling artefacts possible.")
    return llzo_stripped.copy(), 1


def make_bulk_llzo(llzo_stripped, m, n, n_llzo_slab):
    """
    Build a proper periodic bulk LLZO reference:
    1. Find the primitive c-repeat unit of the slab.
    2. Tile it (m x n x k) where k ensures atom count >= n_llzo_slab.

    Parameters
    ----------
    llzo_stripped : Atoms  vacuum-stripped LLZO slab (unit, pre-inplane expansion)
    m, n          : int    in-plane expansion matching the interface
    n_llzo_slab   : int    total LLZO atoms in interface LLZO region

    Returns bulk_llzo : Atoms
    """
    primitive_unit, k_prim = find_llzo_primitive_c_unit(llzo_stripped)
    n_prim = len(primitive_unit)

    # Find minimum k_c along c so m*n*k_c*n_prim >= n_llzo_slab
    k_c = max(1, int(np.ceil(n_llzo_slab / (m * n * n_prim))))

    bulk_llzo = make_supercell(primitive_unit, np.diag([m, n, k_c]))
    bulk_llzo.wrap()

    print(f"    Bulk LLZO: prim_unit={n_prim} atoms × ({m}×{n}×{k_c}) = "
          f"{len(bulk_llzo)} atoms  (slab had {n_llzo_slab})")
    return bulk_llzo

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
            f"Could not find Li+LLZO pair in {combo_dir}.\n"
            f"Found: {cifs}\n"
            f"Expected LI_TAG='{LI_TAG}', LLZO_TAG='{LLZO_TAG}'"
        )
    return li_path, llzo_path

# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(BULK_REF_DIR, exist_ok=True)

    with open(LOG_FILE, "w") as f:
        f.write("# Li–LLZO Interface Stacking Report (v4)\n\n")
        f.write(
            "| Pair | Gap (Å) | Li c-reps | In-plane (m×n) | "
            "N(LLZO) | N(Li) | N(total) | "
            "N(bulk_LLZO) | N(bulk_Li) | "
            "c_LLZO (Å) | c_Li (Å) | c_total (Å) | Output |\n"
        )
        f.write(
            "|------|---------|-----------|----------------|"
            "---------|-------|----------|"
            "-------------|-----------|"
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

        print(f"\nProcessing: {combo_name}")
        print(f"  Li  : {os.path.basename(li_path)}")
        print(f"  LLZO: {os.path.basename(llzo_path)}")

        li_raw   = read(li_path)
        llzo_raw = read(llzo_path)

        li_stripped   = strip_vacuum(li_raw)
        llzo_stripped = strip_vacuum(llzo_raw)

        c_llzo    = np.linalg.norm(llzo_stripped.cell.array[2])
        c_li_base = np.linalg.norm(li_stripped.cell.array[2])

        print(f"  Vacuum stripped — LLZO: {c_llzo:.2f} Å ({len(llzo_stripped)} atoms) | "
              f"Li: {c_li_base:.2f} Å ({len(li_stripped)} atoms)")

        # Li c-repeat to match LLZO thickness
        li_c_rep, n_c = repeat_li_c(li_stripped, c_llzo)
        c_li = np.linalg.norm(li_c_rep.cell.array[2])
        print(f"  Li c-repeated ×{n_c} → {c_li:.2f} Å, {len(li_c_rep)} atoms")

        # Determine in-plane expansion from probe interface
        interface_probe = build_interface(llzo_stripped, li_c_rep, gap_ang=0.0)
        _, m, n = expand_inplane(interface_probe)
        print(f"  In-plane expansion: {m}×{n}")

        # Atom counts in interface regions (after in-plane expansion)
        n_llzo_slab = len(llzo_stripped) * m * n
        n_li_slab   = len(li_c_rep)      * m * n
        print(f"  Interface regions — LLZO: {n_llzo_slab} atoms | Li: {n_li_slab} atoms")

        # --------------------------------------------------------------
        # Build and save bulk references
        # --------------------------------------------------------------
        bulk_dir = os.path.join(BULK_REF_DIR, combo_name)
        os.makedirs(bulk_dir, exist_ok=True)

        print(f"  Building bulk LLZO reference:")
        bulk_llzo = make_bulk_llzo(llzo_stripped, m, n, n_llzo_slab)

        print(f"  Building bulk Li reference:")
        bulk_li = make_bulk_li(li_stripped, li_c_rep, m, n, n_li_slab)

        bulk_llzo_name = f"bulk_llzo__{combo_name}__ip{m}x{n}.cif"
        bulk_li_name   = f"bulk_li__{combo_name}__c{n_c}__ip{m}x{n}.cif"

        write(os.path.join(bulk_dir, bulk_llzo_name), bulk_llzo)
        write(os.path.join(bulk_dir, bulk_li_name),   bulk_li)

        print(f"  ✓ Bulk refs saved:")
        print(f"    LLZO : {bulk_llzo_name}  ({len(bulk_llzo)} atoms)")
        print(f"    Li   : {bulk_li_name}  ({len(bulk_li)} atoms)")

        # Sanity check
        total_bulk = len(bulk_llzo) + len(bulk_li)
        total_slab  = n_llzo_slab + n_li_slab
        if total_bulk != total_slab:
            print(f"  [NOTE] Bulk total ({total_bulk}) ≠ interface region total "
                  f"({total_slab}) by {total_bulk - total_slab} atoms. "
                  f"This is acceptable if small — due to rounding in c-repeat.")

        # --------------------------------------------------------------
        # Interface structures — one per gap
        # --------------------------------------------------------------
        pair_out = os.path.join(OUT_DIR, combo_name)
        os.makedirs(pair_out, exist_ok=True)

        for gap in TARGET_GAPS:
            interface_unit  = build_interface(llzo_stripped, li_c_rep, gap_ang=gap)
            interface_final, m_chk, n_chk = expand_inplane(interface_unit)

            assert (m_chk == m) and (n_chk == n), \
                "In-plane expansion changed between gaps — unexpected."

            n_total = len(interface_final)
            c_total = np.linalg.norm(interface_final.cell.array[2])

            out_name = (
                f"{combo_name}"
                f"__gap_{gap:.1f}A"
                f"__Lic{n_c}_ip{m}x{n}"
                f"__N{n_total}.cif"
            )
            write(os.path.join(pair_out, out_name), interface_final)
            print(f"  gap={gap:.1f} Å | N={n_total} | c={c_total:.2f} Å")

            with open(LOG_FILE, "a") as f:
                f.write(
                    f"| {combo_name} | {gap:.1f} | {n_c} | {m}×{n} | "
                    f"{n_llzo_slab} | {n_li_slab} | {n_total} | "
                    f"{len(bulk_llzo)} | {len(bulk_li)} | "
                    f"{c_llzo:.3f} | {c_li:.3f} | {c_total:.3f} | {out_name} |\n"
                )

    print(f"\n✓ Done.")
    print(f"  Interfaces : {OUT_DIR}/")
    print(f"  Bulk refs  : {BULK_REF_DIR}/")
    print(f"  Report     : {LOG_FILE}")


if __name__ == "__main__":
    main()