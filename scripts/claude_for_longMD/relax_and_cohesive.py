"""
Relax Interface Structures + Cohesive Energy Analysis
======================================================

For each interface CIF in IFACE_DIR:
  1. Relax with MACE (FIRE only, fmax=0.03 eV/Å, max 2000 steps)
  2. Save relaxed CIF + trajectory to RELAX_DIR
  3. Compute cohesive energy using single-atom references

E_coh/atom = (E_relaxed − Σ nᵢ·E_atom_i) / N

Saves:
  RELAX_DIR/  relaxed_<name>.cif
  RELAX_DIR/  traj/<name>.traj
  RELAX_DIR/  cohesive_energy_relaxed_report.md

Author: Mehul Darak et al.
"""

import os
import numpy as np
from collections import Counter
from ase.io import read, write
from ase.optimize import FIRE
from mace.calculators import MACECalculator
import torch

# =============================================================================
# Paths
# =============================================================================

IFACE_DIR  = "/home/mehuldarak/athena/li_and_llzo_stacked_for_long_MD_claude"
RELAX_DIR  = "/home/mehuldarak/athena/li_and_llzo_stacked_for_long_MD_claude_relax"
ATOM_DIR   = "/home/mehuldarak/athena/single_atoms"
MACE_MODEL = (
    "/home/mehuldarak/athena/mace_fps_training/checkpoints/"
    "mace_fps_split17_SaveIT0_256_candidate3.model"
)
REPORT_MD  = os.path.join(RELAX_DIR, "cohesive_energy_relaxed_report.md")
TRAJ_DIR   = os.path.join(RELAX_DIR, "traj")

FMAX_FIRE  = 0.05   # eV/Å
MAX_STEPS  = 2000

# =============================================================================

def main():
    os.makedirs(RELAX_DIR, exist_ok=True)
    os.makedirs(TRAJ_DIR,  exist_ok=True)

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

    # ── Interface CIFs ────────────────────────────────────────────────────────
    iface_cifs = sorted([
        f for f in os.listdir(IFACE_DIR)
        if f.endswith(".cif") and "__longMDstack_v2__" in f
    ])
    print(f"Found {len(iface_cifs)} interface CIFs to relax.\n")

    results = []

    for fname in iface_cifs:
        stem  = fname.replace(".cif", "")
        fpath = os.path.join(IFACE_DIR, fname)

        print(f"{'='*62}")
        print(f"Relaxing: {fname}")

        atoms = read(fpath, index=0)
        N     = len(atoms)
        comp  = dict(Counter(atoms.get_chemical_symbols()))
        print(f"  N={N}  comp={comp}")

        missing = set(atoms.get_chemical_symbols()) - set(e_atom)
        if missing:
            print(f"  [SKIP] missing atom refs: {missing}")
            continue

        traj_path = os.path.join(TRAJ_DIR, f"{stem}.traj")
        out_path  = os.path.join(RELAX_DIR, f"relaxed_{fname}")

        atoms.calc = calc

        # ── FIRE (coarse) ────────────────────────────────────────────────────
        print(f"  FIRE  fmax={FMAX_FIRE} ...")
        fire = FIRE(atoms, trajectory=traj_path, logfile="-")
        fire.run(fmax=FMAX_FIRE, steps=MAX_STEPS)
        print(f"  FIRE done. E = {atoms.get_potential_energy():.4f} eV")

        # ── Save relaxed CIF ─────────────────────────────────────────────────
        write(out_path, atoms)
        print(f"  Saved → {out_path}")

        # ── Cohesive energy ──────────────────────────────────────────────────
        e_relaxed  = atoms.get_potential_energy()

        symbols    = atoms.get_chemical_symbols()
        e_ref      = sum(e_atom[s] for s in symbols)
        e_coh      = e_relaxed - e_ref
        e_coh_atom = e_coh / N

        print(f"  E_coh = {e_coh:.4f} eV  =  {e_coh_atom:.6f} eV/atom\n")

        results.append({
            "name":          stem,
            "N":             N,
            "comp":          comp,
            "e_relaxed":     e_relaxed,
            "e_ref":         e_ref,
            "e_coh":         e_coh,
            "e_coh_per_atom": e_coh_atom,
        })

    # ── Rank ──────────────────────────────────────────────────────────────────
    results.sort(key=lambda r: r["e_coh_per_atom"])

    print(f"\n{'='*62}")
    print("Ranking by relaxed cohesive energy per atom:")
    for i, r in enumerate(results, 1):
        print(f"  {i:2d}. {r['name']}")
        print(f"       {r['e_coh_per_atom']:.6f} eV/atom")

    if results:
        best = results[0]
        print(f"\n★ Most stable (relaxed): {best['name']}")
        print(f"  E_coh/atom = {best['e_coh_per_atom']:.6f} eV/atom")

    # ── Markdown report ───────────────────────────────────────────────────────
    md = (
        "# Relaxed Interface Cohesive Energy Report\n\n"
        f"Model: `{os.path.basename(MACE_MODEL)}`\n\n"
        f"Relaxation: FIRE only, fmax={FMAX_FIRE} eV/Å, max {MAX_STEPS} steps\n\n"
        "```\nE_coh/atom = (E_relaxed − Σ nᵢ·E_atom_i) / N\n```\n\n"
        "**Atomic references:**\n"
    )
    for el, e in sorted(e_atom.items()):
        md += f"- E_atom[{el}] = {e:.6f} eV\n"
    md += (
        "\n| Rank | Structure | N | E_relaxed (eV) | E_ref (eV) | "
        "E_coh (eV) | E_coh/atom (eV/atom) |\n"
        "|------|-----------|---|----------------|------------|"
        "-----------|---------------------|\n"
    )
    for i, r in enumerate(results, 1):
        md += (
            f"| {i} | {r['name']} | {r['N']} | {r['e_relaxed']:.4f} | "
            f"{r['e_ref']:.4f} | {r['e_coh']:.4f} | {r['e_coh_per_atom']:.6f} |\n"
        )
    if results:
        md += (
            f"\n**★ Most stable (relaxed):** {best['name']}  \n"
            f"E_coh/atom = {best['e_coh_per_atom']:.6f} eV/atom\n"
        )

    with open(REPORT_MD, "w") as f:
        f.write(md)

    print(f"\n✓ Report → {REPORT_MD}")


if __name__ == "__main__":
    main()
