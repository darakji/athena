"""
Li Vacancy Creation + MACE Interface Energy Evaluation (v3)
============================================================

Fixes vs v2:
    - Vacancy count uses TOTAL atoms in LLZO region (not just Li count).
      n_vac = round(N_llzo_total / 192) * 2
      This correctly reflects "2 Li vacancies per 192-atom LLZO formula unit".
    - Same fix applied to bulk LLZO vacancy count.
    - Prints a clear vacancy audit line per structure so you can verify.

Interface energy definition:
    E_int = (E_interface_total - E_bulk_LLZO_vac - E_bulk_Li) / A   [eV/Å²]

    E_bulk_LLZO_vac : vacancy-containing LLZO bulk (same vacancy concentration)
    E_bulk_Li       : pristine metallic Li bulk
    A               : interface area = |a × b|

Vacancy consistency:
    - Identify LLZO-side Li by z ∈ [VACUUM_PAD, VACUUM_PAD + c_LLZO]
    - Remove n_vac of them (random, fixed seed=42)
    - Map removed fractional (a,b) coords onto bulk LLZO → remove nearest Li
    - Save vacancy bulk LLZO to BULK_REF_DIR/<combo>/bulk_llzo_vacancy__*.cif

Filters:
    - Only li_100 and li_110 processed
    - li_111 skipped

Author: Mehul Darak et al.
"""

import os
import re
import json
import random
import numpy as np
from ase.io import read, write
from mace.calculators import MACECalculator

# =============================================================================
# Configuration
# =============================================================================

STACKED_DIR     = "/home/mehuldarak/athena/li_and_llzo_stacked_diff_gaps_claude_2000"
BULK_REF_DIR    = "/home/mehuldarak/athena/bulk_references"
MACE_MODEL      = (
    "/home/mehuldarak/athena/mace_fps_training/checkpoints/"
    "mace_fps_split17_SaveIT0_256_candidate3.model"
)

OUT_DIR      = "/home/mehuldarak/athena/li_llzo_best_gap_structures"
LOG_FILE     = "/home/mehuldarak/athena/li_llzo_vacancy_mace_report.md"
RESULTS_JSON = "/home/mehuldarak/athena/li_llzo_vacancy_mace_results.json"

VACUUM_PAD  = 7.5   # Å — must match stacker
RANDOM_SEED = 42

ALLOWED_SURFACES = ["li_100", "li_110"]
SKIP_SURFACES    = ["li_111"]

# =============================================================================
# Filter
# =============================================================================

def should_process(combo_name):
    name_lower = combo_name.lower()
    if any(s in name_lower for s in SKIP_SURFACES):
        return False
    return any(s in name_lower for s in ALLOWED_SURFACES)

# =============================================================================
# LLZO region: all atoms + Li-only indices
# =============================================================================

def get_llzo_region_indices(atoms, c_llzo):
    """
    Return (all_llzo_indices, llzo_li_indices) for atoms in the LLZO slab:
        z_proj ∈ [VACUUM_PAD, VACUUM_PAD + c_llzo]
    """
    cell  = atoms.cell.array
    c_hat = cell[2] / np.linalg.norm(cell[2])
    z_lo  = VACUUM_PAD
    z_hi  = VACUUM_PAD + c_llzo

    all_llzo = []
    llzo_li  = []
    for i, (sym, pos) in enumerate(
        zip(atoms.get_chemical_symbols(), atoms.positions)
    ):
        z = np.dot(pos, c_hat)
        if z_lo <= z <= z_hi:
            all_llzo.append(i)
            if sym == "Li":
                llzo_li.append(i)

    return all_llzo, llzo_li

# =============================================================================
# Infer c_llzo from structure
# =============================================================================

def parse_c_llzo_from_structure(atoms):
    """
    c_LLZO = z-extent of non-Li atoms (exclusively in LLZO slab).
    Returns c_llzo in Å.
    """
    cell  = atoms.cell.array
    c_hat = cell[2] / np.linalg.norm(cell[2])
    non_li_z = [
        np.dot(pos, c_hat)
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.positions)
        if sym != "Li"
    ]
    if not non_li_z:
        raise ValueError("No non-Li atoms found — cannot infer c_LLZO.")
    return max(non_li_z) - min(non_li_z)

# =============================================================================
# Vacancy count
# =============================================================================

def compute_n_vac(n_llzo_total):
    """
    2 Li vacancies per 192 total LLZO atoms.
    n_vac = round(n_llzo_total / 192) * 2
    Minimum of 2.
    """
    return max(2, round(n_llzo_total / 192) * 2)

# =============================================================================
# Vacancy creation — interface
# =============================================================================

def create_interface_vacancies(atoms, llzo_li_indices, n_vac, seed=RANDOM_SEED):
    """
    Remove n_vac Li atoms from the LLZO region of the interface.
    Returns (atoms_with_vacancies, removed_frac_coords).
    removed_frac_coords are used to locate equivalent sites in bulk LLZO.
    """
    if n_vac > len(llzo_li_indices):
        raise ValueError(
            f"Requested {n_vac} vacancies but only {len(llzo_li_indices)} "
            f"Li atoms available in LLZO region."
        )

    rng = random.Random(seed)
    to_remove = sorted(rng.sample(llzo_li_indices, n_vac))

    cell = atoms.cell.array
    removed_frac = [
        np.linalg.solve(cell.T, atoms.positions[idx])
        for idx in to_remove
    ]

    to_keep = [i for i in range(len(atoms)) if i not in to_remove]
    return atoms[to_keep], removed_frac

# =============================================================================
# Vacancy creation — bulk LLZO reference
# =============================================================================

def create_bulk_vacancies(bulk_llzo, removed_frac_interface, n_vac_bulk):
    """
    Remove Li atoms from bulk LLZO at sites equivalent to the interface vacancies.

    Maps fractional (a, b) coordinates of removed interface Li atoms onto bulk,
    finds nearest available Li in bulk, removes it. Periodic boundary aware.

    Returns (atoms_with_vacancies, n_actually_removed).
    """
    bulk_cell = bulk_llzo.cell.array

    bulk_li_indices = [
        i for i, sym in enumerate(bulk_llzo.get_chemical_symbols())
        if sym == "Li"
    ]
    if not bulk_li_indices:
        raise ValueError("No Li atoms in bulk LLZO reference.")

    bulk_li_frac = np.array([
        np.linalg.solve(bulk_cell.T, bulk_llzo.positions[i]) % 1.0
        for i in bulk_li_indices
    ])  # shape (N_li, 3)

    to_remove_bulk = set()

    for frac in removed_frac_interface:
        if len(to_remove_bulk) >= n_vac_bulk:
            break

        fa, fb = frac[0] % 1.0, frac[1] % 1.0

        best_dist      = np.inf
        best_local_idx = None

        for local_idx, global_idx in enumerate(bulk_li_indices):
            if global_idx in to_remove_bulk:
                continue
            fab = bulk_li_frac[local_idx, :2]
            # Periodic in-plane distance
            dfa = min(abs(fa - fab[0]), 1.0 - abs(fa - fab[0]))
            dfb = min(abs(fb - fab[1]), 1.0 - abs(fb - fab[1]))
            dist = np.sqrt(dfa**2 + dfb**2)
            if dist < best_dist:
                best_dist      = dist
                best_local_idx = local_idx

        if best_local_idx is not None:
            to_remove_bulk.add(bulk_li_indices[best_local_idx])

    to_keep = [i for i in range(len(bulk_llzo)) if i not in to_remove_bulk]
    return bulk_llzo[to_keep], len(to_remove_bulk)

# =============================================================================
# Interface energy
# =============================================================================

def interface_energy(e_total, e_bulk_llzo_vac, e_bulk_li, area):
    """E_int = (E_total - E_bulk_LLZO_vac - E_bulk_Li) / A  [eV/Å²]"""
    return (e_total - e_bulk_llzo_vac - e_bulk_li) / area

# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading MACE model...")
    calc = MACECalculator(
        model_paths=MACE_MODEL, device="cuda", default_dtype="float64"
    )
    print("MACE model loaded.\n")

    with open(LOG_FILE, "w") as f:
        f.write("# Li–LLZO Vacancy + MACE Interface Energy Report (v3)\n\n")
        f.write(
            "## Vacancy audit\n"
            "n_vac = round(N_llzo_total_atoms / 192) * 2\n\n"
        )
        f.write(
            "| Combo | Gap (Å) | N_llzo_total | N_llzo_Li | "
            "n_vac | N_bulk_llzo_total | n_vac_bulk | "
            "E_total (eV) | E_bulk_LLZO_vac (eV) | E_bulk_Li (eV) | "
            "E_int (eV/Å²) | Area (Å²) |\n"
        )
        f.write(
            "|-------|---------|--------------|-----------|"
            "-------|-------------------|------------|"
            "-------------|---------------------|----------------|"
            "--------------|----------|\n"
        )

    combo_dirs = sorted([
        d for d in os.listdir(STACKED_DIR)
        if os.path.isdir(os.path.join(STACKED_DIR, d))
    ])

    results = {}

    for combo_name in combo_dirs:
        if not should_process(combo_name):
            print(f"[SKIP] {combo_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {combo_name}")

        # -- Bulk refs --
        bulk_dir = os.path.join(BULK_REF_DIR, combo_name)
        if not os.path.isdir(bulk_dir):
            print(f"  [SKIP] No bulk ref dir: {bulk_dir}")
            continue

        bulk_files  = os.listdir(bulk_dir)
        bulk_llzo_f = next(
            (f for f in bulk_files
             if f.startswith("bulk_llzo") and "vacancy" not in f), None
        )
        bulk_li_f = next(
            (f for f in bulk_files if f.startswith("bulk_li")), None
        )

        if bulk_llzo_f is None or bulk_li_f is None:
            print(f"  [SKIP] Missing bulk ref files in {bulk_dir}")
            continue

        bulk_llzo_pristine = read(os.path.join(bulk_dir, bulk_llzo_f))
        bulk_li            = read(os.path.join(bulk_dir, bulk_li_f))

        bulk_li.calc = calc
        e_bulk_li    = bulk_li.get_potential_energy()
        print(f"  E_bulk_Li = {e_bulk_li:.4f} eV  ({len(bulk_li)} atoms)")

        n_bulk_llzo_total = len(bulk_llzo_pristine)
        n_bulk_llzo_li    = sum(
            1 for s in bulk_llzo_pristine.get_chemical_symbols() if s == "Li"
        )
        n_vac_bulk = compute_n_vac(n_bulk_llzo_total)
        print(f"  Bulk LLZO: {n_bulk_llzo_total} total atoms, "
              f"{n_bulk_llzo_li} Li → {n_vac_bulk} vacancies to create")

        # -- Interface CIFs --
        combo_stacked_dir = os.path.join(STACKED_DIR, combo_name)
        cif_files = sorted(
            f for f in os.listdir(combo_stacked_dir) if f.endswith(".cif")
        )

        gap_results    = []
        bulk_vac_cache = {}   # n_vac_bulk -> (e_bulk_llzo_vac, n_actually_removed)

        for cif_name in cif_files:
            gap_match = re.search(r"__gap_([\d.]+)A__", cif_name)
            if gap_match is None:
                print(f"  [WARN] Cannot parse gap from {cif_name}, skipping.")
                continue
            gap = float(gap_match.group(1))

            atoms = read(os.path.join(combo_stacked_dir, cif_name))

            try:
                c_llzo = parse_c_llzo_from_structure(atoms)
            except ValueError as e:
                print(f"  [WARN] {cif_name}: {e}")
                continue

            # Identify LLZO region
            all_llzo_idx, llzo_li_idx = get_llzo_region_indices(atoms, c_llzo)
            n_llzo_total = len(all_llzo_idx)
            n_llzo_li    = len(llzo_li_idx)
            n_vac        = compute_n_vac(n_llzo_total)

            # ── VACANCY AUDIT ──────────────────────────────────────────────
            print(f"  gap={gap:.1f} Å | LLZO region: {n_llzo_total} total atoms, "
                  f"{n_llzo_li} Li | n_vac = round({n_llzo_total}/192)*2 = {n_vac}")
            # ───────────────────────────────────────────────────────────────

            if n_vac > n_llzo_li:
                print(f"  [WARN] n_vac ({n_vac}) > n_llzo_li ({n_llzo_li}). "
                      f"Clamping to {n_llzo_li}.")
                n_vac = n_llzo_li

            # Create interface vacancies
            atoms_vac, removed_frac = create_interface_vacancies(
                atoms, llzo_li_idx, n_vac
            )

            area = np.linalg.norm(
                np.cross(atoms_vac.cell.array[0], atoms_vac.cell.array[1])
            )

            # Vacancy bulk LLZO (cached — same for all gaps)
            if n_vac_bulk not in bulk_vac_cache:
                bulk_llzo_vac, n_actually_removed = create_bulk_vacancies(
                    bulk_llzo_pristine, removed_frac, n_vac_bulk
                )
                vac_bulk_name = (
                    f"bulk_llzo_vacancy__{combo_name}"
                    f"__nvac{n_actually_removed}.cif"
                )
                write(os.path.join(bulk_dir, vac_bulk_name), bulk_llzo_vac)
                print(f"  Saved vacancy bulk LLZO → {vac_bulk_name}")

                bulk_llzo_vac.calc = calc
                e_bulk_llzo_vac    = bulk_llzo_vac.get_potential_energy()
                print(f"  E_bulk_LLZO_vac = {e_bulk_llzo_vac:.4f} eV  "
                      f"({len(bulk_llzo_vac)} atoms, {n_actually_removed} vac)")

                bulk_vac_cache[n_vac_bulk] = (e_bulk_llzo_vac, n_actually_removed)
            else:
                e_bulk_llzo_vac, n_actually_removed = bulk_vac_cache[n_vac_bulk]

            # MACE energy
            atoms_vac.calc = calc
            e_total = atoms_vac.get_potential_energy()
            e_int   = interface_energy(e_total, e_bulk_llzo_vac, e_bulk_li, area)

            print(f"  → E_int = {e_int:.6f} eV/Å²  |  Area = {area:.2f} Å²")

            gap_results.append({
                "gap":             gap,
                "e_interface":     e_int,
                "e_total":         e_total,
                "e_bulk_llzo_vac": e_bulk_llzo_vac,
                "e_bulk_li":       e_bulk_li,
                "area":            area,
                "n_vac":           n_vac,
                "n_vac_bulk":      n_actually_removed,
                "n_llzo_total":    n_llzo_total,
                "n_llzo_li":       n_llzo_li,
                "cif_name":        cif_name,
                "atoms_vac":       atoms_vac,
            })

            with open(LOG_FILE, "a") as f:
                f.write(
                    f"| {combo_name} | {gap:.1f} | {n_llzo_total} | {n_llzo_li} | "
                    f"{n_vac} | {n_bulk_llzo_total} | {n_actually_removed} | "
                    f"{e_total:.4f} | {e_bulk_llzo_vac:.4f} | {e_bulk_li:.4f} | "
                    f"{e_int:.6f} | {area:.2f} |\n"
                )

        if not gap_results:
            print(f"  [SKIP] No valid results for {combo_name}.")
            continue

        best = min(gap_results, key=lambda x: x["e_interface"])
        print(f"\n  ★ Best gap: {best['gap']:.1f} Å  "
              f"(E_int = {best['e_interface']:.6f} eV/Å²)")

        best_out_name = (
            f"BEST__{combo_name}"
            f"__gap_{best['gap']:.1f}A"
            f"__Eint_{best['e_interface']:.5f}eVA2"
            f"__Nvac{best['n_vac']}.cif"
        )
        write(os.path.join(OUT_DIR, best_out_name), best["atoms_vac"])
        print(f"  Saved → {best_out_name}")

        results[combo_name] = {
            "best_gap":         best["gap"],
            "best_e_interface": best["e_interface"],
            "best_file":        best_out_name,
            "n_vac_interface":  best["n_vac"],
            "n_vac_bulk":       best["n_vac_bulk"],
            "all_gaps": [
                {k: v for k, v in r.items() if k != "atoms_vac"}
                for r in gap_results
            ],
        }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Done.")
    print(f"  Best structures  : {OUT_DIR}/")
    print(f"  Vacancy bulk refs: {BULK_REF_DIR}/<combo>/bulk_llzo_vacancy__*.cif")
    print(f"  Report           : {LOG_FILE}")
    print(f"  JSON             : {RESULTS_JSON}")


if __name__ == "__main__":
    main()