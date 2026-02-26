# ============================================================
# run_md_active_region_all_cifs.py
# ============================================================

import os, sys, glob, time
import numpy as np

from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.constraints import FixAtoms
from ase import units
from mace.calculators import MACECalculator


# ============================================================
# PATHS
# ============================================================

IN_BASE = "/eagle/DFTCalculations/mehul/ml/athena/polaris/li_llzo_relaxed_bestgaps_polaris"
OUT_BASE = "/eagle/DFTCalculations/mehul/ml/athena/polaris/li_llzo_md_polaris"

CIF_OUT  = os.path.join(OUT_BASE, "cifs")
TRAJ_OUT = os.path.join(OUT_BASE, "traj")
LOG_OUT  = os.path.join(OUT_BASE, "logs")

os.makedirs(CIF_OUT, exist_ok=True)
os.makedirs(TRAJ_OUT, exist_ok=True)
os.makedirs(LOG_OUT, exist_ok=True)


# ============================================================
# MACE CALCULATOR
# ============================================================

MODEL_PATH = "/eagle/DFTCalculations/mehul/ml/MACE_models/2023-12-03-mace-128-L1_epoch-199.model"

calc = MACECalculator(
    model_paths=MODEL_PATH,
    device="cuda",
    default_dtype="float32",
)


# ============================================================
# MD PARAMETERS
# ============================================================

TEMPERATURES = [300, 600]      # K
TIMESTEP_FS = 1.0
NSTEPS_MD = 5000
SNAPSHOT_INTERVAL = 25

FRICTION = 0.01 / units.fs


# ============================================================
# INPUT CIF COLLECTION (RECURSIVE)
# ============================================================

if len(sys.argv) > 1:
    cif_files = sys.argv[1:]
else:
    cif_files = sorted(
        glob.glob(os.path.join(IN_BASE, "**", "*.cif"), recursive=True)
    )

print(f"Found {len(cif_files)} CIF files", flush=True)


# ============================================================
# MAIN LOOP
# ============================================================

for cif_file in cif_files:
    name = os.path.splitext(os.path.basename(cif_file))[0]
    start_time = time.time()

    for T in TEMPERATURES:

        # ----------------------------------------------------
        # LOAD STRUCTURE FRESH
        # ----------------------------------------------------
        atoms = read(cif_file)
        atoms.set_pbc([True, True, False])
        atoms.calc = calc

        natoms = len(atoms)

        # ----------------------------------------------------
        # DEFINE FROZEN / ACTIVE REGIONS
        # ----------------------------------------------------
        symbols = np.array(atoms.get_chemical_symbols())
        z = atoms.positions[:, 2]

        li_mask = symbols == "Li"
        llzo_mask = ~li_mask

        # Freeze bottom Li reservoir
        freeze_li = li_mask & (z < np.percentile(z[li_mask], 20))

        # Freeze top LLZO reservoir
        freeze_llzo = llzo_mask & (z > np.percentile(z[llzo_mask], 80))

        freeze_mask = freeze_li | freeze_llzo
        atoms.set_constraint(FixAtoms(mask=freeze_mask))

        # ----------------------------------------------------
        # VELOCITY INITIALIZATION (ACTIVE ATOMS ONLY)
        # ----------------------------------------------------
        MaxwellBoltzmannDistribution(atoms, T * units.kB)
        Stationary(atoms)
        ZeroRotation(atoms)

        vel = atoms.get_velocities()
        vel[freeze_mask] = 0.0
        atoms.set_velocities(vel)

        # ----------------------------------------------------
        # OUTPUTS
        # ----------------------------------------------------
        traj = Trajectory(
            os.path.join(TRAJ_OUT, f"{name}_T{T}K.traj"),
            "w",
            atoms,
        )

        dyn = Langevin(
            atoms,
            TIMESTEP_FS * units.fs,
            T * units.kB,
            FRICTION,
            logfile=os.path.join(LOG_OUT, f"{name}_T{T}K.log"),
        )

        step = {"i": 0}

        def write_outputs():
            traj.write(atoms)
            step["i"] += 1

            if step["i"] % SNAPSHOT_INTERVAL == 0:
                write(
                    os.path.join(
                        CIF_OUT,
                        f"{name}_T{T}K_{step['i']}.cif",
                    ),
                    atoms,
                )

        dyn.attach(write_outputs, interval=1)

        try:
            dyn.run(NSTEPS_MD)
        except Exception as e:
            print(f"[SKIP] {name} @ {T} K → {e}", flush=True)
        finally:
            traj.close()

    elapsed = time.time() - start_time
    with open(os.path.join(OUT_BASE, "MD_TIME_ESTIMATE.md"), "a") as f:
        f.write(
            f"- {name}: {natoms} atoms | "
            f"{elapsed/60:.2f} min | "
            f"{elapsed/natoms:.4f} s/atom\n"
        )

print("ALL MD FINISHED")