"""
Long-MD Interface Energy Evaluation — Grand Canonical Formula
==============================================================

Grand canonical interface energy:

    delta_n_Li = n_Li(interface) − [n_Li(Li_bulk) + n_Li(LLZO_pristine_bulk)]

    gamma = [ E_interface − E_Li_bulk − E_LLZO_pristine_bulk
              − delta_n_Li × μ_Li ]  /  (2 A)

    delta_n_Li < 0  because LLZO has vacancies, so the interface has fewer
    Li than the sum of the two pristine bulk references.

Also computed for reference:

    gamma_ref = (E_interface − E_LLZO_vac − E_Li_bulk) / (2 A)

The /2A factor accounts for two equivalent interfaces per periodic cell.

Bulk files read from BULK_DIR/<combo>/:
    bulk_llzo_Nx_vacancy__*.cif   — LLZO with vacancies
    bulk_llzo_Nx_pristine__*.cif  — LLZO pristine
    bulk_li_Mx__*.cif             — Li pristine

Author: Mehul Darak et al.
"""

import os
import glob
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

# =============================================================================
# Paths & constants
# =============================================================================

IFACE_DIR  = "/home/mehuldarak/athena/li_and_llzo_stacked_for_long_MD_claude"
BULK_DIR   = "/home/mehuldarak/athena/bulk_with_vacacnies_claudeWritten"
MACE_MODEL = (
    "/home/mehuldarak/athena/mace_fps_training/checkpoints/"
    "mace_fps_split17_SaveIT0_256_candidate3.model"
)
REPORT_MD  = os.path.join(IFACE_DIR, "interface_energy_report.md")

MU_LI      = -192.6    # eV / Li atom   (chemical potential reference)
EV_TO_J    = 1.60218e-19
ANG2_TO_M2 = 1e-20

# =============================================================================
# Helpers
# =============================================================================

def combo_from_iface_name(fname):
    stem  = fname.replace(".cif", "")
    parts = stem.split("__longMDstack_v2__")
    if len(parts) != 2:
        raise ValueError(f"Cannot parse combo from: {fname}")
    return f"{parts[0]}__{parts[1]}"


def find_bulk_files(combo_name):
    combo_bulk = os.path.join(BULK_DIR, combo_name)
    if not os.path.isdir(combo_bulk):
        raise FileNotFoundError(f"Bulk dir not found: {combo_bulk}")

    def _first(pattern, label):
        matches = glob.glob(os.path.join(combo_bulk, pattern))
        if not matches:
            raise FileNotFoundError(f"No {label} in {combo_bulk}")
        return matches[0]

    return (
        _first("bulk_llzo_*x_vacancy__*.cif",  "bulk LLZO vacancy"),
        _first("bulk_llzo_*x_pristine__*.cif", "bulk LLZO pristine"),
        _first("bulk_li_*x__*.cif",            "bulk Li"),
    )


def n_li(atoms):
    return sum(1 for s in atoms.get_chemical_symbols() if s == "Li")


def area_of(atoms):
    c = atoms.cell.array
    return float(np.linalg.norm(np.cross(c[0], c[1])))


def to_jm2(e_ev, a_ang2):
    return (e_ev * EV_TO_J) / (2.0 * a_ang2 * ANG2_TO_M2)


# =============================================================================
# Main
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

    iface_cifs = sorted([
        f for f in os.listdir(IFACE_DIR)
        if f.endswith(".cif") and "__longMDstack_v2__" in f
    ])
    print(f"Found {len(iface_cifs)} interface CIFs.\n")

    md_rows = []
    md_header = (
        "# Long-MD Interface Energy Report — Grand Canonical\n\n"
        f"Model: `{os.path.basename(MACE_MODEL)}`  \n"
        f"μ_Li = **{MU_LI} eV/atom**\n\n"
        "```\n"
        "delta_n_Li = n_Li(interface) − [n_Li(Li_bulk) + n_Li(LLZO_pristine)]\n"
        "gamma = (E_interface − E_Li − E_LLZO_prist − delta_n_Li × μ_Li) / (2A)\n"
        "```\n\n"
        "| Combo | A (Å²) | n_Li_iface | n_Li_Li | n_Li_LLZO | Δn_Li | "
        "E_iface (eV) | E_LLZO_prist (eV) | E_Li (eV) | "
        "γ (eV/Å²) | γ (J/m²) | γ_ref (J/m²) |\n"
        "|-------|--------|-----------|---------|----------|-------|"
        "-------------|------------------|-----------|"
        "-----------|---------|-------------|\n"
    )

    results = []

    for iface_fname in iface_cifs:
        iface_path = os.path.join(IFACE_DIR, iface_fname)
        print(f"{'='*62}")
        print(f"Interface : {iface_fname}")

        try:
            combo_name = combo_from_iface_name(iface_fname)
        except ValueError as e:
            print(f"  [SKIP] {e}"); continue

        try:
            llzo_vac_path, llzo_prist_path, li_path = find_bulk_files(combo_name)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}"); continue

        print(f"  LLZO vac   : {os.path.basename(llzo_vac_path)}")
        print(f"  LLZO prist : {os.path.basename(llzo_prist_path)}")
        print(f"  Li bulk    : {os.path.basename(li_path)}")

        # ── Read structures ──────────────────────────────────────────────────
        iface      = read(iface_path)
        llzo_vac   = read(llzo_vac_path)
        llzo_prist = read(llzo_prist_path)
        bulk_li    = read(li_path)

        # ── Li atom counts ───────────────────────────────────────────────────
        n_li_iface = n_li(iface)
        n_li_li    = n_li(bulk_li)
        n_li_llzo  = n_li(llzo_prist)
        delta_n_li = n_li_iface - (n_li_li + n_li_llzo)

        A = area_of(iface)

        print(f"  n_Li : iface={n_li_iface}  Li_bulk={n_li_li}  "
              f"LLZO_prist={n_li_llzo}  Δn_Li={delta_n_li}")
        print(f"  Area : {A:.3f} Å²")

        # ── MACE energies ────────────────────────────────────────────────────
        iface.calc      = calc;  e_iface      = iface.get_potential_energy()
        llzo_prist.calc = calc;  e_llzo_prist = llzo_prist.get_potential_energy()
        llzo_vac.calc   = calc;  e_llzo_vac   = llzo_vac.get_potential_energy()
        bulk_li.calc    = calc;  e_li         = bulk_li.get_potential_energy()

        print(f"  E_iface      = {e_iface:.4f} eV  ({len(iface)} atoms)")
        print(f"  E_LLZO_prist = {e_llzo_prist:.4f} eV  ({len(llzo_prist)} atoms)")
        print(f"  E_LLZO_vac   = {e_llzo_vac:.4f} eV  ({len(llzo_vac)} atoms)")
        print(f"  E_Li         = {e_li:.4f} eV  ({len(bulk_li)} atoms)")

        # ── Grand canonical gamma ─────────────────────────────────────────────
        dE = e_iface - e_li - e_llzo_prist - delta_n_li * MU_LI
        g_ev  = dE / (2.0 * A)
        g_si  = to_jm2(dE, A)

        # ── Reference gamma (vacancy bulk) ────────────────────────────────────
        dE_ref   = e_iface - e_llzo_vac - e_li
        g_ref_si = to_jm2(dE_ref, A)

        print(f"  Δn_Li × μ_Li = {delta_n_li} × {MU_LI} = {delta_n_li * MU_LI:.2f} eV")
        print(f"  γ  (grand canonical) = {g_ev:.6f} eV/Å²  =  {g_si:.4f} J/m²")
        print(f"  γ_ref (vac bulk)     = {g_ref_si:.4f} J/m²\n")

        results.append({
            "combo":   combo_name,
            "A":       A,
            "n_li_iface": n_li_iface, "n_li_li":   n_li_li,
            "n_li_llzo":  n_li_llzo,  "delta_n_li": delta_n_li,
            "e_iface":    e_iface,    "e_llzo_prist": e_llzo_prist,
            "e_li":       e_li,
            "g_ev":    g_ev,  "g_si":    g_si,
            "g_ref_si": g_ref_si,
        })

        md_rows.append(
            f"| {combo_name} | {A:.2f} | {n_li_iface} | {n_li_li} | "
            f"{n_li_llzo} | {delta_n_li} | {e_iface:.4f} | {e_llzo_prist:.4f} | "
            f"{e_li:.4f} | {g_ev:.6f} | {g_si:.4f} | {g_ref_si:.4f} |\n"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_lines = []
    if results:
        best = min(results, key=lambda r: r["g_si"])
        summary_lines = [
            "\n## Summary\n\n",
            f"**Most stable (γ grand canonical):** {best['combo']}  \n",
            f"γ = {best['g_si']:.4f} J/m²  ({best['g_ev']:.6f} eV/Å²)\n\n",
            "---\n",
            f"*μ_Li = {MU_LI} eV/atom | Model: {os.path.basename(MACE_MODEL)}*\n",
        ]
        print(f"{'='*62}")
        print(f"★ Most stable: {best['combo']}")
        print(f"  γ = {best['g_si']:.4f} J/m²  ({best['g_ev']:.6f} eV/Å²)")

    # ── Write MD ─────────────────────────────────────────────────────────────
    with open(REPORT_MD, "w") as f:
        f.write(md_header)
        f.writelines(md_rows)
        f.writelines(summary_lines)

    print(f"\n✓ Report → {REPORT_MD}")


if __name__ == "__main__":
    main()
