"""
Li–LLZO Interface Stacker v2 — Long MD Structures
===================================================

Workflow (per combo, ALL surfaces including li_111):
  1. Read LLZO slab + Li slab from MATCHED_DIR (already lattice-matched).
  2. Strip vacuum from both.
  3. Remove n_vac = round(N_llzo / 192) * 2  Li atoms from LLZO unit (seed=42).
     Applied per LLZO repeat → identical vacancy pattern in every repeat.
  4. Stack vacancy-LLZO × N_LLZO  (no vacuum) → save as bulk ref.
  5. Stack pristine Li × N_LI     (no vacuum) → save as bulk ref.
  6. Build interface:
        | 7.5 Å | LLZO×N_LLZO (vac) | 2 Å gap | Li×N_LI | 7.5 Å |
  7. Write per-combo atom-count table to Markdown in OUT_DIR.

Repeat counts (by surface):
  li_100, li_110  →  LLZO×3, Li×5
  li_111          →  LLZO×2, Li×3

Outputs:
  OUT_DIR/  *.cif                        — interface structures
  OUT_DIR/  stacking_report.md           — atom-count table
  BULK_DIR/ <combo>/bulk_llzo_Nx_vacancy__nvac<N>.cif
  BULK_DIR/ <combo>/bulk_li_Mx.cif

Author: Mehul Darak et al.
"""

import os
import random
import numpy as np
from ase.io import read, write
from ase import Atoms

# =============================================================================
# Configuration
# =============================================================================

MATCHED_DIR = "/home/mehuldarak/athena/li_and_llzo_unrelaxed_seperate"
OUT_DIR     = "/home/mehuldarak/athena/li_and_llzo_stacked_for_long_MD_claude"
BULK_DIR    = "/home/mehuldarak/athena/bulk_with_vacacnies_claudeWritten"

GAP        = 2.0    # Å — interface gap
VACUUM_PAD = 10    # Å — vacuum on each outer face

LI_TAG   = "__scaled_to_"
LLZO_TAG = "__fixed.cif"

RANDOM_SEED = 42

# Repeat counts per surface family
REPEATS = {
    "li_111": (2, 3),   # (N_LLZO, N_LI)
    "default": (3, 5),
}

# =============================================================================
# Helpers
# =============================================================================

def get_repeats(combo_name):
    """Return (n_llzo_repeats, n_li_repeats) based on surface in combo name."""
    name_lower = combo_name.lower()
    for key, val in REPEATS.items():
        if key in name_lower:
            return val
    return REPEATS["default"]


def strip_vacuum(atoms):
    """Remove vacuum padding; return slab with atoms snug inside cell."""
    atoms  = atoms.copy()
    cell   = atoms.cell.array.copy()
    c_len  = np.linalg.norm(cell[2])
    c_hat  = cell[2] / c_len
    frac_z = np.array([np.dot(pos, c_hat) / c_len for pos in atoms.positions])

    z_min, z_max = frac_z.min(), frac_z.max()
    atoms.positions -= (z_min - 0.001) * cell[2]

    new_c_frac  = (z_max - z_min) + 0.003
    new_cell    = cell.copy()
    new_cell[2] = c_hat * (new_c_frac * c_len)
    atoms.set_cell(new_cell, scale_atoms=False)
    atoms.wrap()
    return atoms


# =============================================================================
# Vacancy creation
# =============================================================================

def compute_n_vac(n_llzo_total):
    """2 Li vacancies per 192 LLZO atoms; minimum 2."""
    return max(2, round(n_llzo_total / 192) * 2)


def remove_vacancies(atoms, n_vac, seed=RANDOM_SEED):
    """Randomly remove n_vac Li atoms from `atoms`."""
    li_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Li"]
    if n_vac > len(li_indices):
        raise ValueError(
            f"Requested {n_vac} vacancies but only {len(li_indices)} Li atoms present."
        )
    rng       = random.Random(seed)
    to_remove = set(rng.sample(li_indices, n_vac))
    to_keep   = [i for i in range(len(atoms)) if i not in to_remove]
    return atoms[to_keep]


# =============================================================================
# Stacking (no vacuum)
# =============================================================================

def stack_repeat(atoms, n):
    """
    Stack `atoms` n times along the c-axis with NO vacuum between repeats.
    `atoms` must be vacuum-stripped so cell[2] == slab thickness.
    """
    cell   = atoms.cell.array.copy()
    c_hat  = cell[2] / np.linalg.norm(cell[2])
    c_unit = np.linalg.norm(cell[2])

    new_positions = []
    new_symbols   = []

    for k in range(n):
        offset = k * c_unit * c_hat
        new_positions.append(atoms.positions + offset)
        new_symbols.extend(atoms.get_chemical_symbols())

    new_cell    = cell.copy()
    new_cell[2] = c_hat * (n * c_unit)

    return Atoms(
        symbols   = new_symbols,
        positions = np.vstack(new_positions),
        cell      = new_cell,
        pbc       = True,
    )


# =============================================================================
# Interface builder
# =============================================================================

def build_interface(llzo_Nx, li_Mx, gap=GAP, vac=VACUUM_PAD):
    """
    Build:  | vac | LLZO×N | gap | Li×M | vac |
    Both inputs must be vacuum-stripped (cell[2] = slab thickness only).
    """
    cell_llzo = llzo_Nx.cell.array
    c_llzo    = np.linalg.norm(cell_llzo[2])
    c_li      = np.linalg.norm(li_Mx.cell.array[2])
    c_hat     = cell_llzo[2] / c_llzo

    c_total  = vac + c_llzo + gap + c_li + vac

    new_cell    = np.zeros((3, 3))
    new_cell[0] = cell_llzo[0]
    new_cell[1] = cell_llzo[1]
    new_cell[2] = c_hat * c_total

    llzo_pos = llzo_Nx.positions.copy() + vac * c_hat
    li_pos   = li_Mx.positions.copy()   + (vac + c_llzo + gap) * c_hat

    return Atoms(
        symbols   = list(llzo_Nx.get_chemical_symbols()) + list(li_Mx.get_chemical_symbols()),
        positions = np.vstack([llzo_pos, li_pos]),
        cell      = new_cell,
        pbc       = True,
    )


# =============================================================================
# File discovery
# =============================================================================

def find_pair(combo_dir):
    cifs      = [f for f in os.listdir(combo_dir) if f.endswith(".cif")]
    li_path   = None
    llzo_path = None
    for f in cifs:
        if LI_TAG in f:
            li_path   = os.path.join(combo_dir, f)
        if f.endswith(LLZO_TAG):
            llzo_path = os.path.join(combo_dir, f)
    if li_path is None or llzo_path is None:
        raise ValueError(f"Could not find pair in {combo_dir}. Found: {cifs}")
    return li_path, llzo_path


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUT_DIR,  exist_ok=True)
    os.makedirs(BULK_DIR, exist_ok=True)

    md_path = os.path.join(OUT_DIR, "stacking_report.md")

    combo_dirs = sorted([
        d for d in os.listdir(MATCHED_DIR)
        if os.path.isdir(os.path.join(MATCHED_DIR, d))
    ])

    print(f"Found {len(combo_dirs)} combo(s) in {MATCHED_DIR}\n")

    # ── Markdown table header ────────────────────────────────────────────────
    md_rows = []
    md_header = (
        "# Long-MD Interface Stacking Report\n\n"
        "Vacancy rule: **2 Li removed per LLZO repeat** "
        f"(= round(N_llzo / 192) × 2, seed={RANDOM_SEED})\n\n"
        "Repeat counts: li_100 / li_110 → LLZO×3, Li×5 | li_111 → LLZO×2, Li×3\n\n"
        "Interface layout: `| 7.5 Å | LLZO×N (vac) | 2 Å | Li×M | 7.5 Å |`\n\n"
        "| Combo | N_LLZO rep | N_Li rep | LLZO unit atoms | Li unit atoms | "
        "n_vac/rep | LLZO×N atoms | Li×M atoms | "
        "Interface total | c_LLZO×N (Å) | c_Li×M (Å) | c_total (Å) |\n"
        "|-------|-----------|---------|----------------|--------------|"
        "----------|-------------|-----------|"
        "----------------|-------------|-----------|-------------|\n"
    )

    for combo_name in combo_dirs:
        combo_path = os.path.join(MATCHED_DIR, combo_name)
        try:
            li_path, llzo_path = find_pair(combo_path)
        except ValueError as e:
            print(f"[SKIP] {e}")
            continue

        n_llzo_rep, n_li_rep = get_repeats(combo_name)

        # ── Read & strip vacuum ──────────────────────────────────────────────
        llzo_raw = read(llzo_path)
        li_raw   = read(li_path)
        llzo     = strip_vacuum(llzo_raw)
        li       = strip_vacuum(li_raw)

        n_llzo_unit = len(llzo)
        n_li_unit   = len(li)

        # ── Vacancy removal (per LLZO unit) ──────────────────────────────────
        n_vac = compute_n_vac(n_llzo_unit)
        llzo_vac        = remove_vacancies(llzo, n_vac)
        n_llzo_vac_unit = len(llzo_vac)

        # ── Stack LLZO×N and Li×M ────────────────────────────────────────────
        llzo_Nx         = stack_repeat(llzo_vac, n_llzo_rep)   # with vacancies
        llzo_Nx_pristine = stack_repeat(llzo,    n_llzo_rep)   # no vacancies
        li_Mx           = stack_repeat(li,       n_li_rep)

        n_llzo_Nx = len(llzo_Nx)
        n_li_Mx   = len(li_Mx)

        # ── a/b sanity check ─────────────────────────────────────────────────
        a_li   = np.linalg.norm(li.cell.array[0])
        b_li   = np.linalg.norm(li.cell.array[1])
        a_llzo = np.linalg.norm(llzo.cell.array[0])
        b_llzo = np.linalg.norm(llzo.cell.array[1])
        da, db = abs(a_li - a_llzo), abs(b_li - b_llzo)
        if da > 0.05 or db > 0.05:
            print(f"[WARN] {combo_name}: a/b mismatch Δa={da:.4f} Å  Δb={db:.4f} Å")

        # ── Save bulk LLZO: vacancy + pristine (N× repeated) ────────────────
        combo_bulk_dir = os.path.join(BULK_DIR, combo_name)
        os.makedirs(combo_bulk_dir, exist_ok=True)

        n_vac_total = n_llzo_rep * n_vac
        bulk_llzo_name = (
            f"bulk_llzo_{n_llzo_rep}x_vacancy__{combo_name}"
            f"__nvac{n_vac_total}.cif"
        )
        bulk_llzo_path = os.path.join(combo_bulk_dir, bulk_llzo_name)

        # pristine (no vacancies)
        bulk_llzo_pristine_name = f"bulk_llzo_{n_llzo_rep}x_pristine__{combo_name}.cif"
        bulk_llzo_pristine_path = os.path.join(combo_bulk_dir, bulk_llzo_pristine_name)
        write(bulk_llzo_path, llzo_Nx)
        write(bulk_llzo_pristine_path, llzo_Nx_pristine)

        # ── Save bulk Li (M× repeated, pristine) ─────────────────────────────
        bulk_li_name = f"bulk_li_{n_li_rep}x__{combo_name}.cif"
        bulk_li_path = os.path.join(combo_bulk_dir, bulk_li_name)
        write(bulk_li_path, li_Mx)

        # ── Build interface ───────────────────────────────────────────────────
        interface = build_interface(llzo_Nx, li_Mx)
        n_total   = len(interface)

        c_llzo_Nx = np.linalg.norm(llzo_Nx.cell.array[2])
        c_li_Mx   = np.linalg.norm(li_Mx.cell.array[2])
        c_total   = np.linalg.norm(interface.cell.array[2])

        # ── Output name ───────────────────────────────────────────────────────
        li_tag   = combo_name.split("__")[0]
        llzo_tag = combo_name.split("__", 1)[1]
        out_name = f"{li_tag}__longMDstack_v2__{llzo_tag}.cif"
        out_path = os.path.join(OUT_DIR, out_name)
        write(out_path, interface)

        # ── Console log ───────────────────────────────────────────────────────
        print(f"[OK] {out_name}")
        print(f"     Repeats        : LLZO×{n_llzo_rep}, Li×{n_li_rep}")
        print(f"     LLZO unit      : {n_llzo_unit} atoms  →  {n_vac} Li removed  →  {n_llzo_vac_unit} / repeat")
        print(f"     LLZO×{n_llzo_rep:<2} (vac)  : {n_llzo_Nx} atoms  (c = {c_llzo_Nx:.3f} Å)  → {bulk_llzo_path}")
        print(f"     LLZO×{n_llzo_rep:<2} (prist): {len(llzo_Nx_pristine)} atoms  → {bulk_llzo_pristine_path}")
        print(f"     Li×{n_li_rep:<5}         : {n_li_Mx} atoms   (c = {c_li_Mx:.3f} Å)  → {bulk_li_path}")
        print(f"     Interface      : {n_total} atoms total  (c_total = {c_total:.3f} Å)\n")

        # ── Markdown row ─────────────────────────────────────────────────────
        md_rows.append(
            f"| {combo_name} | {n_llzo_rep} | {n_li_rep} | {n_llzo_unit} | {n_li_unit} | "
            f"{n_vac} | {n_llzo_Nx} | {n_li_Mx} | {n_total} | "
            f"{c_llzo_Nx:.3f} | {c_li_Mx:.3f} | {c_total:.3f} |\n"
        )

    # ── Write Markdown ────────────────────────────────────────────────────────
    with open(md_path, "w") as f:
        f.write(md_header)
        f.writelines(md_rows)
        f.write(
            f"\n---\n"
            f"*Gap: {GAP} Å | Vacuum pad: {VACUUM_PAD} Å | "
            f"Vacancy seed: {RANDOM_SEED}*\n"
        )

    print(f"✓ Done.")
    print(f"  Interfaces     → {OUT_DIR}/")
    print(f"  Bulk LLZO vac  → {BULK_DIR}/<combo>/bulk_llzo_Nx_vacancy__*.cif")
    print(f"  Bulk Li        → {BULK_DIR}/<combo>/bulk_li_Mx__*.cif")
    print(f"  MD report      → {md_path}")


if __name__ == "__main__":
    main()
