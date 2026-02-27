# ============================================================
# MD with geometrically defined active interface region
# (EXACT same logic as relaxation script)
# ============================================================

import os
import glob
import time
import numpy as np

from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase import units

from mace.calculators import MACECalculator


# =======================
# Paths
# =======================

# in_dir = "/home/mehuldarak/athena/polaris/scripts/remaining_slab"
# out_base = "/home/mehuldarak/athena/polaris/scripts/remaining_slab_md"

in_dir = "/eagle/DFTCalculations/mehul/ml/athena/polaris/li_llzo_relaxed_bestgaps_polaris/cifs"
out_base = "/eagle/DFTCalculations/mehul/ml/athena/polaris/li_llzo_md_polaris"

cif_out  = os.path.join(out_base, "cifs")
traj_out = os.path.join(out_base, "traj")
log_out  = os.path.join(out_base, "logs")

os.makedirs(cif_out, exist_ok=True)
os.makedirs(traj_out, exist_ok=True)
os.makedirs(log_out, exist_ok=True)


# =======================
# MACE model
# =======================

# model_path = "/home/mehuldarak/MACE_models/universal_09072025/2024-01-07-mace-128-L2_epoch-199.model"
model_path = "/eagle/DFTCalculations/mehul/ml/MACE_models/2023-12-03-mace-128-L1_epoch-199.model"

print("Loading MACE calculator...", flush=True)

calc = MACECalculator(
    model_paths=model_path,
    device="cuda",
    default_dtype="float32",
)


# =======================
# MD parameters
# =======================

TEMPERATURES = [300, 600]    # K
TIMESTEP_FS = 1.0
NSTEPS_MD = 5000
SNAPSHOT_INTERVAL = 25

FRICTION = 0.01 / units.fs

active_fraction = 0.85
min_active_angstrom = 6.0

# =======================
# Collect CIFs
# =======================

import sys

if len(sys.argv) > 1:
    cif_files = [
        os.path.abspath(arg)
        for arg in sys.argv[1:]
        if arg.strip()
    ]
else:
    # fallback: run all CIFs in directory
    cif_files = sorted(glob.glob(os.path.join(in_dir, "*.cif")))

print("Running MD on the following CIFs:", flush=True)
for c in cif_files:
    if not c.endswith(".cif") or not os.path.isfile(c):
        raise RuntimeError(f"Invalid CIF input: {c}")

# =======================
# Main loop
# =======================

for cif_file in cif_files:

    fname = os.path.basename(cif_file)
    base_name = os.path.splitext(fname)[0]
    start_time = time.time()

    for T in TEMPERATURES:

        print(f"\nMD: {fname} @ {T} K", flush=True)

        # -----------------------
        # Load structure fresh
        # -----------------------
        atoms = read(cif_file)
        atoms.set_pbc([True, True, False])
        atoms.calc = calc

        symbols = np.array(atoms.get_chemical_symbols())
        z = atoms.positions[:, 2]

        # -----------------------
        # Identify slabs (SAME LOGIC)
        # -----------------------
        oxide_mask = np.isin(symbols, ["La", "Zr", "O"])
        if not np.any(oxide_mask):
            raise RuntimeError("No oxide atoms found – cannot identify LLZO slab.")

        llzo_top = z[oxide_mask].max()

        li_metal_mask = (symbols == "Li") & (z > llzo_top)
        if not np.any(li_metal_mask):
            raise RuntimeError("No Li-metal slab found above LLZO.")

        llzo_mask = oxide_mask

        llzo_z = z[llzo_mask]
        li_z   = z[li_metal_mask]

        llzo_bottom, llzo_top = llzo_z.min(), llzo_z.max()
        li_bottom, li_top     = li_z.min(), li_z.max()

        # -----------------------
        # Active region thickness
        # -----------------------
        llzo_thickness = llzo_top - llzo_bottom
        li_thickness   = li_top - li_bottom

        llzo_active_thickness = max(
            active_fraction * llzo_thickness,
            min_active_angstrom,
        )
        li_active_thickness = max(
            active_fraction * li_thickness,
            min_active_angstrom,
        )

        llzo_active_min = llzo_top - llzo_active_thickness
        li_active_max   = li_bottom + li_active_thickness

        # -----------------------
        # Freeze mask (SAME LOGIC)
        # -----------------------
        freeze_mask = (
            (llzo_mask & (z < llzo_active_min)) |
            (li_metal_mask & (z > li_active_max))
        )

        atoms.set_constraint(FixAtoms(mask=freeze_mask))

        print(
            f"Frozen atoms: {np.sum(freeze_mask)} / {len(atoms)} | "
            f"LLZO frozen: {np.sum(llzo_mask & freeze_mask)} / {np.sum(llzo_mask)} | "
            f"Li frozen: {np.sum(li_metal_mask & freeze_mask)} / {np.sum(li_metal_mask)}",
            flush=True
        )

        # -----------------------
        # Velocities (ACTIVE ONLY)
        # -----------------------
        MaxwellBoltzmannDistribution(atoms, T * units.kB)
        Stationary(atoms)
        ZeroRotation(atoms)

        vel = atoms.get_velocities()
        vel[freeze_mask] = 0.0
        atoms.set_velocities(vel)

        # -----------------------
        # Outputs
        # -----------------------
        traj_path = os.path.join(traj_out, f"{base_name}_T{T}K.traj")
        log_path  = os.path.join(log_out, f"{base_name}_T{T}K.log")

        traj = Trajectory(traj_path, "w", atoms)

        dyn = Langevin(
            atoms,
            TIMESTEP_FS * units.fs,
            T * units.kB,
            FRICTION,
            logfile=log_path,
        )

        step = {"i": 0}

        def write_outputs():
            atoms.wrap(pbc=[True, True, False])   # <<< THIS LINE
            traj.write(atoms)
            step["i"] += 1

            if step["i"] % SNAPSHOT_INTERVAL == 0:
                write(
                    os.path.join(
                        cif_out,
                        f"{base_name}_T{T}K_{step['i']}.cif",
                    ),
                    atoms,
                )

        dyn.attach(write_outputs, interval=1)

        try:
            dyn.run(NSTEPS_MD)
        finally:
            traj.close()

    elapsed = time.time() - start_time
    print(
        f"Finished {fname} in {elapsed/60:.2f} min",
        flush=True
    )

print("\nALL MD FINISHED")