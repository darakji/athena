import os
from ase.io import read, write
from ase.optimize import FIRE
from mace.calculators import MACECalculator

# ==================================================
# Paths
# ==================================================
IN_DIR  = "/home/mehuldarak/athena/li_llzo_unrelaxed_bestgaps"
OUT_DIR = "/home/mehuldarak/athena/li_llzo_relaxed_bestgaps"

CIF_DIR  = os.path.join(OUT_DIR, "cifs")
LOG_DIR  = os.path.join(OUT_DIR, "logs")
TRAJ_DIR = os.path.join(OUT_DIR, "traj")

os.makedirs(CIF_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)

# ==================================================
# MACE calculator
# ==================================================
calc = MACECalculator(
    model_paths=[
        "/home/mehuldarak/MACE_models/universal_09072025/"
        "2023-12-03-mace-128-L1_epoch-199.model"
    ],
    device="cuda",      # switch to "cpu" if needed
    default_dtype="float32",
)

# ==================================================
# Relaxation parameters
# ==================================================
FMAX = 0.05

# ==================================================
# Loop over best-gap CIFs
# ==================================================
for fname in sorted(os.listdir(IN_DIR)):

    if not fname.endswith(".cif"):
        continue
    if fname == "best_gap_summary.md":
        continue

    in_cif = os.path.join(IN_DIR, fname)
    print(f"\nRelaxing: {fname}")

    atoms = read(in_cif)
    atoms.calc = calc

    base = fname.replace(".cif", "")
    out_cif  = os.path.join(CIF_DIR,  base + "_r.cif")
    out_log  = os.path.join(LOG_DIR,  base + ".log")
    out_traj = os.path.join(TRAJ_DIR, base + ".traj")

    dyn = FIRE(
        atoms,
        logfile=out_log,
        trajectory=out_traj
    )
    dyn.run(fmax=FMAX)

    write(out_cif, atoms)

    print(f"  → relaxed CIF : {out_cif}")
    print(f"  → log         : {out_log}")
    print(f"  → trajectory  : {out_traj}")

print("\n✔ All best-gap structures relaxed successfully.")