"""
Post-process Relaxation Trajectories
=====================================

For each trajectory in TRAJ_DIR:
  1. Scan *every* frame and find the one with the lowest fmax
     (max force magnitude on any atom, exact same criterion that FIRE uses).
  2. Delete the existing relaxed_<name>.cif in RELAX_DIR.
  3. Write the best frame as the new relaxed_<name>.cif at the same path.
  4. Compute cohesive energy from atomic references.

Report columns:
  Structure | N | fmax (eV/Å) | E_relaxed (eV) | E_ref (eV) | E_coh (eV) | E_coh/atom (eV/atom)

Author: auto-generated
"""

import os
import numpy as np
from collections import Counter
from ase.io import read, write
from ase.io.trajectory import Trajectory
from mace.calculators import MACECalculator
import torch

# =============================================================================
# Paths  — same as relax_and_cohesive.py
# =============================================================================

RELAX_DIR  = "/home/mehuldarak/athena/li_and_llzo_stacked_for_long_MD_claude_relax"
ATOM_DIR   = "/home/mehuldarak/athena/single_atoms"
MACE_MODEL = (
    "/home/mehuldarak/athena/mace_fps_training/checkpoints/"
    "mace_fps_split17_SaveIT0_256_candidate3.model"
)
TRAJ_DIR   = os.path.join(RELAX_DIR, "traj")
REPORT_MD  = os.path.join(RELAX_DIR, "best_frame_cohesive_report.md")

# =============================================================================


def fmax_of_frame(atoms):
    """Return the maximum force magnitude (eV/Å) for an ASE Atoms object.
    Forces must already be attached (read from traj with calc info)."""
    forces = atoms.get_forces()          # shape (N, 3)
    return float(np.max(np.linalg.norm(forces, axis=1)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading MACE model...")
    calc = MACECalculator(
        model_paths=MACE_MODEL,
        device=device,
        default_dtype="float32",
        batch_size=16,
    )
    print("MACE model loaded.\n")

    # ── Atomic reference energies ─────────────────────────────────────────────
    print("Computing atomic references...")
    e_atom = {}
    for fname in sorted(os.listdir(ATOM_DIR)):
        if not fname.endswith(".cif"):
            continue
        el   = fname.replace(".cif", "")
        atom = read(os.path.join(ATOM_DIR, fname), index=0)
        atom.calc = calc
        e_atom[el] = atom.get_potential_energy()
        print(f"  E_atom[{el}] = {e_atom[el]:.6f} eV")
    print()

    # ── Trajectory files ──────────────────────────────────────────────────────
    traj_files = sorted(
        f for f in os.listdir(TRAJ_DIR) if f.endswith(".traj")
    )
    print(f"Found {len(traj_files)} trajectory files.\n")

    results = []

    for traj_fname in traj_files:
        stem      = traj_fname.replace(".traj", "")
        traj_path = os.path.join(TRAJ_DIR, traj_fname)
        cif_path  = os.path.join(RELAX_DIR, f"relaxed_{stem}.cif")

        print(f"{'='*70}")
        print(f"Processing: {stem}")

        # ── Read all frames ───────────────────────────────────────────────────
        try:
            traj = Trajectory(traj_path, "r")
            n_frames = len(traj)
            print(f"  Frames in trajectory: {n_frames}")
        except Exception as exc:
            print(f"  [ERROR] Cannot open trajectory: {exc}")
            continue

        if n_frames == 0:
            print("  [SKIP] Empty trajectory.")
            continue

        # ── Find frame with lowest fmax ───────────────────────────────────────
        best_idx   = None
        best_fmax  = np.inf
        best_atoms = None

        for idx in range(n_frames):
            frame = traj[idx]
            try:
                fm = fmax_of_frame(frame)
            except Exception:
                # forces not stored — should not happen with FIRE traj
                continue
            if fm < best_fmax:
                best_fmax  = fm
                best_idx   = idx
                best_atoms = frame

        traj.close()

        if best_atoms is None:
            print("  [SKIP] No frame with forces found.")
            continue

        print(f"  Best frame: idx={best_idx}  fmax={best_fmax:.6f} eV/Å")

        # ── Replace the saved CIF ─────────────────────────────────────────────
        if os.path.exists(cif_path):
            os.remove(cif_path)
            print(f"  Deleted old CIF: {os.path.basename(cif_path)}")

        write(cif_path, best_atoms)
        print(f"  Saved best-frame CIF → {os.path.basename(cif_path)}")

        # ── Cohesive energy ───────────────────────────────────────────────────
        N       = len(best_atoms)
        symbols = best_atoms.get_chemical_symbols()
        comp    = dict(Counter(symbols))

        missing = set(symbols) - set(e_atom)
        if missing:
            print(f"  [WARN] Missing atom refs: {missing} — skipping cohesive E")
            e_relaxed = e_ref = e_coh = e_coh_atom = float("nan")
        else:
            e_relaxed  = best_atoms.get_potential_energy()
            e_ref      = sum(e_atom[s] for s in symbols)
            e_coh      = e_relaxed - e_ref
            e_coh_atom = e_coh / N

        print(f"  N={N}  comp={comp}")
        print(f"  E_relaxed = {e_relaxed:.6f} eV")
        print(f"  E_coh     = {e_coh:.6f} eV   E_coh/atom = {e_coh_atom:.8f} eV/atom\n")

        results.append({
            "name":           stem,
            "N":              N,
            "comp":           comp,
            "best_frame_idx": best_idx,
            "fmax":           best_fmax,
            "e_relaxed":      e_relaxed,
            "e_ref":          e_ref,
            "e_coh":          e_coh,
            "e_coh_per_atom": e_coh_atom,
        })

    # ── Sort by E_coh/atom (most stable first) ────────────────────────────────
    results.sort(key=lambda r: r["e_coh_per_atom"])

    print(f"\n{'='*70}")
    print("Ranking by E_coh/atom (most stable → least stable):")
    for i, r in enumerate(results, 1):
        print(f"  {i:2d}. {r['name']}")
        print(f"       fmax = {r['fmax']:.6f} eV/Å  |  "
              f"E_coh/atom = {r['e_coh_per_atom']:.8f} eV/atom  |  "
              f"E_coh = {r['e_coh']:.4f} eV")

    if results:
        best = results[0]
        print(f"\n★ Most stable: {best['name']}")
        print(f"  fmax        = {best['fmax']:.6f} eV/Å")
        print(f"  E_coh/atom  = {best['e_coh_per_atom']:.8f} eV/atom")
        print(f"  E_coh       = {best['e_coh']:.6f} eV")

    # ── Markdown report ───────────────────────────────────────────────────────
    md  = "# Best-Frame Cohesive Energy Report\n\n"
    md += f"Model: `{os.path.basename(MACE_MODEL)}`\n\n"
    md += (
        "For each structure the trajectory frame with the **lowest fmax** "
        "(maximum atomic force magnitude) was selected, saved as the relaxed CIF, "
        "and used for cohesive energy analysis.\n\n"
    )
    md += "```\nE_coh/atom = (E_relaxed − Σ nᵢ·E_atom_i) / N\n```\n\n"
    md += "**Atomic references:**\n"
    for el, e in sorted(e_atom.items()):
        md += f"- E_atom[{el}] = {e:.6f} eV\n"
    md += "\n"

    # Table header
    md += (
        "| Rank | Structure | N | Best Frame | fmax (eV/Å) | "
        "E_relaxed (eV) | E_ref (eV) | E_coh (eV) | E_coh/atom (eV/atom) |\n"
        "|------|-----------|---|------------|-------------|"
        "----------------|------------|------------|----------------------|\n"
    )
    for i, r in enumerate(results, 1):
        md += (
            f"| {i} | {r['name']} | {r['N']} | {r['best_frame_idx']} | "
            f"{r['fmax']:.6f} | {r['e_relaxed']:.6f} | {r['e_ref']:.6f} | "
            f"{r['e_coh']:.6f} | {r['e_coh_per_atom']:.8f} |\n"
        )

    if results:
        best = results[0]
        md += (
            f"\n**★ Most stable:** {best['name']}  \n"
            f"fmax = {best['fmax']:.6f} eV/Å  \n"
            f"E_coh/atom = {best['e_coh_per_atom']:.8f} eV/atom  \n"
            f"E_coh = {best['e_coh']:.6f} eV\n"
        )

    with open(REPORT_MD, "w") as f:
        f.write(md)

    print(f"\n✓ Report → {REPORT_MD}")


if __name__ == "__main__":
    main()
