"""
Bulk-with-Vacancies Relaxation
================================

For every CIF in every subdirectory of SRC_DIR:
  1. Add 10 Å vacuum on both sides in z (via ASE surface.add_vacuum)
  2. Relax with MACE — FIRE only, fmax=0.05 eV/Å, max 2000 steps
  3. Save trajectory alongside the output CIF
  4. After all steps complete, scan the trajectory and pick the frame
     with the lowest fmax, write that as <stem>_r.cif in the mirrored
     subdirectory under OUT_DIR.
  5. Per-structure log written as <stem>_r.log in the same output subdir.

Directory layout preserved:
  SRC_DIR/<subdir>/<stem>.cif
  OUT_DIR/<subdir>/<stem>_r.cif        ← lowest-fmax frame
  OUT_DIR/<subdir>/<stem>_r.traj
  OUT_DIR/<subdir>/<stem>_r.log

Author: auto-generated
"""

import os
import sys
import numpy as np
from collections import Counter
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from mace.calculators import MACECalculator
import torch

# =============================================================================
# Config
# =============================================================================

SRC_DIR    = "/home/mehuldarak/athena/bulk_with_vacacnies_claudeWritten"
OUT_DIR    = "/home/mehuldarak/athena/bulk_with_vacacnies_claudeWritten_relaxed"
MACE_MODEL = (
    "/home/mehuldarak/athena/mace_fps_training/checkpoints/"
    "mace_fps_split17_SaveIT0_256_candidate3.model"
)

VACUUM_ANG = 10.0   # Å added on each side in z
FMAX       = 0.05   # eV/Å
MAX_STEPS  = 2000

# =============================================================================


def add_vacuum_both_sides(atoms, vacuum=VACUUM_ANG):
    """
    Add `vacuum` Å of vacuum on BOTH sides of the slab in z.
    Strategy:
      - centre atoms in z
      - extend cell c-vector by 2*vacuum
      - shift atoms so they sit in the middle of the new cell
    Works even for tilted cells because we operate on fractional
    coordinates along the c-axis only.
    """
    from ase import Atoms as _Atoms
    cell = atoms.cell.copy()             # 3×3
    pos  = atoms.get_positions().copy()

    # Current slab thickness in Cartesian z
    z_min = pos[:, 2].min()
    z_max = pos[:, 2].max()
    thickness = z_max - z_min

    # New cell c-length = thickness + 2*vacuum
    # (assumes c is mostly along z — true for all three Li-slab orientations)
    c_hat    = cell[2] / np.linalg.norm(cell[2])   # unit vector of c
    c_length = np.linalg.norm(cell[2])
    new_c_length = thickness + 2 * vacuum

    cell[2] = c_hat * new_c_length

    # Shift atoms so z_min aligns with vacuum boundary
    z_shift = vacuum - z_min
    pos[:, 2] += z_shift

    atoms.set_cell(cell)
    atoms.set_positions(pos)
    atoms.pbc = [True, True, True]
    return atoms


def fmax_of_frame(atoms):
    forces = atoms.get_forces()
    return float(np.max(np.linalg.norm(forces, axis=1)))


def best_frame_from_traj(traj_path):
    traj   = Trajectory(traj_path, "r")
    n      = len(traj)
    best_fm, best_idx, best_atoms = np.inf, None, None
    for i in range(n):
        frame = traj[i]
        try:
            fm = fmax_of_frame(frame)
        except Exception:
            continue
        if fm < best_fm:
            best_fm, best_idx, best_atoms = fm, i, frame
    traj.close()
    return best_idx, best_fm, best_atoms


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Loading MACE model...", flush=True)
    calc = MACECalculator(
        model_paths=MACE_MODEL,
        device=device,
        default_dtype="float32",
        batch_size=16,
    )
    print("MACE model loaded.\n", flush=True)

    # Discover all CIFs
    tasks = []   # list of (subdir_name, cif_filename)
    for subdir in sorted(os.listdir(SRC_DIR)):
        subdir_path = os.path.join(SRC_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in sorted(os.listdir(subdir_path)):
            if fname.endswith(".cif"):
                tasks.append((subdir, fname))

    print(f"Found {len(tasks)} CIF files across {len(set(t[0] for t in tasks))} subdirectories.\n",
          flush=True)

    results = []

    for subdir, fname in tasks:
        stem     = fname.replace(".cif", "")
        src_path = os.path.join(SRC_DIR, subdir, fname)

        out_subdir = os.path.join(OUT_DIR, subdir)
        os.makedirs(out_subdir, exist_ok=True)

        out_cif  = os.path.join(out_subdir, f"{stem}_r.cif")
        out_traj = os.path.join(out_subdir, f"{stem}_r.traj")
        out_log  = os.path.join(out_subdir, f"{stem}_r.log")

        sep = "=" * 72
        print(sep, flush=True)
        print(f"Structure : {subdir}/{fname}", flush=True)

        # ── Read & add vacuum ────────────────────────────────────────────────
        atoms = read(src_path, index=0)
        N     = len(atoms)
        comp  = dict(Counter(atoms.get_chemical_symbols()))
        print(f"  Atoms: {N}  Composition: {comp}", flush=True)

        z_before = np.linalg.norm(atoms.cell[2])
        atoms    = add_vacuum_both_sides(atoms, VACUUM_ANG)
        z_after  = np.linalg.norm(atoms.cell[2])
        print(f"  Cell c: {z_before:.3f} Å → {z_after:.3f} Å  (+{z_after-z_before:.1f} Å vacuum)", flush=True)

        # ── Attach calculator ────────────────────────────────────────────────
        atoms.calc = calc

        # ── FIRE relaxation ──────────────────────────────────────────────────
        print(f"  FIRE  fmax={FMAX} eV/Å  max_steps={MAX_STEPS}", flush=True)
        with open(out_log, "w") as logfile:
            optimizer = FIRE(atoms, trajectory=out_traj, logfile=logfile)
            optimizer.run(fmax=FMAX, steps=MAX_STEPS)

        converged = optimizer.get_number_of_steps() < MAX_STEPS
        print(f"  FIRE done.  Steps={optimizer.get_number_of_steps()}  "
              f"converged={converged}", flush=True)

        # ── Pick lowest-fmax frame ───────────────────────────────────────────
        best_idx, best_fm, best_atoms = best_frame_from_traj(out_traj)

        if best_atoms is None:
            print("  [WARN] No frames with forces in traj — using final atoms.", flush=True)
            best_atoms = atoms
            best_fm    = fmax_of_frame(atoms)
            best_idx   = optimizer.get_number_of_steps()

        print(f"  Best frame: idx={best_idx}  fmax={best_fm:.6f} eV/Å", flush=True)

        # ── Write output CIF ─────────────────────────────────────────────────
        write(out_cif, best_atoms)
        print(f"  Saved → {out_cif}", flush=True)

        e_relaxed = best_atoms.get_potential_energy()
        print(f"  E = {e_relaxed:.6f} eV\n", flush=True)

        results.append({
            "subdir":    subdir,
            "stem":      stem,
            "N":         N,
            "comp":      comp,
            "best_idx":  best_idx,
            "fmax":      best_fm,
            "converged": converged,
            "e_relaxed": e_relaxed,
            "out_cif":   out_cif,
        })

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*72}", flush=True)
    print("SUMMARY — all structures (sorted by fmax):", flush=True)
    results.sort(key=lambda r: r["fmax"])
    for i, r in enumerate(results, 1):
        tag = "✓" if r["converged"] else "✗"
        print(f"  {i:2d}. [{tag}] {r['subdir']}/{r['stem']}_r", flush=True)
        print(f"       fmax={r['fmax']:.6f} eV/Å  E={r['e_relaxed']:.4f} eV  "
              f"N={r['N']}", flush=True)

    # ── Markdown report ─────────────────────────────────────────────────────
    report_path = os.path.join(OUT_DIR, "relaxation_report.md")
    md  = "# Bulk-with-Vacancies Relaxation Report\n\n"
    md += f"Model: `{os.path.basename(MACE_MODEL)}`\n\n"
    md += f"Vacuum added: {VACUUM_ANG} Å each side (z-direction)\n\n"
    md += f"Relaxation: FIRE, fmax={FMAX} eV/Å, max {MAX_STEPS} steps\n\n"
    md += "Saved frame: **lowest fmax** across the full trajectory\n\n"
    md += (
        "| # | Subdir | Stem | N | Converged | Best Frame | fmax (eV/Å) | E_relaxed (eV) |\n"
        "|---|--------|------|---|-----------|------------|-------------|----------------|\n"
    )
    for i, r in enumerate(results, 1):
        c = "Yes" if r["converged"] else "No"
        md += (
            f"| {i} | {r['subdir']} | {r['stem']}_r | {r['N']} | {c} | "
            f"{r['best_idx']} | {r['fmax']:.6f} | {r['e_relaxed']:.6f} |\n"
        )
    with open(report_path, "w") as f:
        f.write(md)
    print(f"\n✓ Report → {report_path}", flush=True)


if __name__ == "__main__":
    main()
