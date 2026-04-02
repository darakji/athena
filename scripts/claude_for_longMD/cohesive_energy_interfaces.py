"""
Cohesive Energy of Interface Structures
========================================

E_coh/atom = (E_structure - Σ n_i * E_atom_i) / N_total

Atomic references from: /home/mehuldarak/athena/single_atoms/
Interface structures from: li_and_llzo_stacked_for_long_MD_claude/
"""

import os
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

IFACE_DIR   = "/home/mehuldarak/athena/li_and_llzo_stacked_for_long_MD_claude"
ATOM_DIR    = "/home/mehuldarak/athena/single_atoms"
MACE_MODEL  = (
    "/home/mehuldarak/athena/mace_fps_training/checkpoints/"
    "mace_fps_split17_SaveIT0_256_candidate3.model"
)
REPORT_MD   = os.path.join(IFACE_DIR, "cohesive_energy_report.md")

# =============================================================================

def main():
    print("Loading MACE model...")
    calc = MACECalculator(
        model_paths=MACE_MODEL,
        device="cuda",
        default_dtype="float32",
        batch_size=16,
    )
    print("MACE model loaded.\n")

    # ── Single-atom reference energies ───────────────────────────────────────
    print("Computing atomic reference energies...")
    e_atom = {}
    for fname in sorted(os.listdir(ATOM_DIR)):
        if not fname.endswith(".cif"):
            continue
        element = fname.replace(".cif", "")
        atom = read(os.path.join(ATOM_DIR, fname), index=0)
        atom.calc = calc
        e = atom.get_potential_energy()
        e_atom[element] = e
        print(f"  E_atom[{element}] = {e:.6f} eV")

    print()

    # ── Interface CIFs ───────────────────────────────────────────────────────
    iface_cifs = sorted([
        f for f in os.listdir(IFACE_DIR)
        if f.endswith(".cif") and "__longMDstack_v2__" in f
    ])
    print(f"Found {len(iface_cifs)} interface CIFs.\n")

    results = []

    for fname in iface_cifs:
        fpath = os.path.join(IFACE_DIR, fname)
        atoms = read(fpath, index=0)
        symbols = atoms.get_chemical_symbols()
        N = len(atoms)

        # Reference energy sum
        missing = set(symbols) - set(e_atom)
        if missing:
            print(f"[SKIP] {fname}: missing atom refs for {missing}")
            continue

        e_ref = sum(e_atom[s] for s in symbols)

        atoms.calc = calc
        e_total = atoms.get_potential_energy()

        e_coh_total = e_total - e_ref          # eV
        e_coh_per_atom = e_coh_total / N       # eV/atom

        from collections import Counter
        comp = dict(Counter(symbols))
        print(f"[OK] {fname}")
        print(f"     N={N}  comp={comp}")
        print(f"     E_total={e_total:.4f} eV  E_ref={e_ref:.4f} eV")
        print(f"     E_coh = {e_coh_total:.4f} eV  = {e_coh_per_atom:.6f} eV/atom\n")

        results.append({
            "name":          fname.replace(".cif", ""),
            "N":             N,
            "comp":          comp,
            "e_total":       e_total,
            "e_ref":         e_ref,
            "e_coh":         e_coh_total,
            "e_coh_per_atom": e_coh_per_atom,
        })

    # ── Rank ──────────────────────────────────────────────────────────────────
    results.sort(key=lambda r: r["e_coh_per_atom"])

    print("=" * 62)
    print("Ranking by cohesive energy per atom (most negative = most stable):")
    for i, r in enumerate(results, 1):
        print(f"  {i:2d}. {r['name']}")
        print(f"       {r['e_coh_per_atom']:.6f} eV/atom")

    best = results[0]
    print(f"\n★ Most stable: {best['name']}")
    print(f"  E_coh/atom = {best['e_coh_per_atom']:.6f} eV/atom")

    # ── Markdown ─────────────────────────────────────────────────────────────
    md = (
        "# Interface Cohesive Energy Report\n\n"
        f"Model: `{os.path.basename(MACE_MODEL)}`\n\n"
        "```\nE_coh/atom = (E_structure − Σ nᵢ·E_atom_i) / N\n```\n\n"
        "**Atomic references:**\n"
    )
    for el, e in sorted(e_atom.items()):
        md += f"- E_atom[{el}] = {e:.6f} eV\n"
    md += (
        "\n| Rank | Structure | N | E_total (eV) | E_ref (eV) | "
        "E_coh (eV) | E_coh/atom (eV/atom) |\n"
        "|------|-----------|---|-------------|------------|"
        "-----------|---------------------|\n"
    )
    for i, r in enumerate(results, 1):
        md += (
            f"| {i} | {r['name']} | {r['N']} | {r['e_total']:.4f} | "
            f"{r['e_ref']:.4f} | {r['e_coh']:.4f} | {r['e_coh_per_atom']:.6f} |\n"
        )
    md += (
        f"\n**Most stable:** {best['name']}  \n"
        f"E_coh/atom = {best['e_coh_per_atom']:.6f} eV/atom\n"
    )

    with open(REPORT_MD, "w") as f:
        f.write(md)

    print(f"\n✓ Report → {REPORT_MD}")


if __name__ == "__main__":
    main()
