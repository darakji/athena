import os
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE
from ase.constraints import FixAtoms
from mace.calculators import MACECalculator

# ==================================================
# Paths
# ==================================================
in_base  = "/home/mehuldarak/athena/li_and_llzo_unrelaxed_seperate"
out_base = "/home/mehuldarak/athena/li_and_llzo_relaxed_seperate"

os.makedirs(out_base, exist_ok=True)

# ==================================================
# MACE calculator (FILL MODEL PATH)
# ==================================================
calc = MACECalculator(
    model_paths=["/home/mehuldarak/MACE_models/universal_09072025/2023-12-03-mace-128-L1_epoch-199.model"],   # ← FILL THIS
    device="cuda",      # or "cpu"
    default_dtype="float32",
)

# ==================================================
# Parameters
# ==================================================
FMAX = 0.05
Z_TOL = 0.5  # Å, tolerance to identify surface layer

# ==================================================
# Helper: apply layer constraints
# ==================================================
def apply_layer_constraint(atoms, mode):
    """
    mode = 'freeze_top' or 'freeze_bottom'
    """
    z = atoms.positions[:, 2]
    z_min, z_max = z.min(), z.max()

    if mode == "freeze_top":
        mask = z > (z_max - Z_TOL)
    elif mode == "freeze_bottom":
        mask = z < (z_min + Z_TOL)
    else:
        raise ValueError("Unknown constraint mode")

    atoms.set_constraint(FixAtoms(indices=np.where(mask)[0]))

# ==================================================
# Main directory walk
# ==================================================
for root, _, files in os.walk(in_base):

    rel_path = os.path.relpath(root, in_base)
    out_dir = os.path.join(out_base, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    for fname in files:
        if not fname.endswith(".cif"):
            continue

        in_cif  = os.path.join(root, fname)
        out_cif = os.path.join(out_dir, fname)

        atoms = read(in_cif)
        atoms.calc = calc

        # ------------------------------------------
        # Decide constraint based on material
        # ------------------------------------------
        name_lower = fname.lower()

        if "li" in name_lower and "llzo" not in name_lower:
            apply_layer_constraint(atoms, "freeze_top")
            constraint_info = "Li: top layer frozen"

        elif "llzo" in name_lower:
            apply_layer_constraint(atoms, "freeze_bottom")
            constraint_info = "LLZO: bottom layer frozen"

        else:
            raise RuntimeError(f"Cannot classify slab: {fname}")

        # ------------------------------------------
        # Relax
        # ------------------------------------------
        log_path = out_cif.replace(".cif", ".log")

        dyn = FIRE(atoms, logfile=log_path)
        dyn.run(fmax=FMAX)

        write(out_cif, atoms)

        print(f"Relaxed: {out_cif}")
        print(f"  → {constraint_info}")

print("✔ All slabs relaxed with preserved directory structure.")