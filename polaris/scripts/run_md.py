# ============================================================
# run_md_xy_pbc_zwall.py
# ============================================================

import os, sys, glob, time
import numpy as np

from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    ZeroRotation,
    Stationary,
)
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import MACECalculator


# ============================================================
# HARD WALL CALCULATOR (z confinement)
# ============================================================

class ZWallCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, z_lo, z_hi, k_wall):
        super().__init__()
        self.base = base_calc
        self.z_lo = z_lo
        self.z_hi = z_hi
        self.k = k_wall

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        self.base.calculate(atoms, properties, system_changes)

        forces = self.base.results["forces"].copy()
        energy = self.base.results.get("energy", 0.0)

        z = atoms.positions[:, 2]

        # lower wall
        mask = z < self.z_lo
        if np.any(mask):
            dz = self.z_lo - z[mask]
            forces[mask, 2] += self.k * dz
            energy += 0.5 * self.k * np.sum(dz ** 2)

        # upper wall
        mask = z > self.z_hi
        if np.any(mask):
            dz = z[mask] - self.z_hi
            forces[mask, 2] -= self.k * dz
            energy += 0.5 * self.k * np.sum(dz ** 2)

        self.results["forces"] = forces
        self.results["energy"] = energy


# ============================================================
# PATHS
# ============================================================

IN_DIR = "/eagle/DFTCalculations/mehul/ml/athena/polaris/li_llzo_relaxed_bestgaps_polaris/cifs"
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

MODEL_PATH = "/eagle/DFTCalculations/mehul/ml/MACE_models/2024-01-07-mace-128-L2_epoch-199.model"

BASE_CALC = MACECalculator(
    model_paths=MODEL_PATH,
    device="cuda",
    default_dtype="float32",
)


# ============================================================
# MD PARAMETERS
# ============================================================

TEMPERATURES = [300, 600]      # K
TIMESTEP_FS = 1.0              # fs (reduce to 0.5 if unstable)
NSTEPS_MD = 5000
SNAPSHOT_INTERVAL = 25

FRICTION = 0.03 / units.fs
VACUUM_PADDING = 8.0           # Å
K_WALL = 500.0                 # eV / Å^2  (soft but safe)


# ============================================================
# INPUT SELECTION
# ============================================================

if len(sys.argv) > 1:
    cif_files = [os.path.join(IN_DIR, f) for f in sys.argv[1:]]
else:
    cif_files = sorted(glob.glob(os.path.join(IN_DIR, "*.cif")))

print(f"Found {len(cif_files)} structures", flush=True)


# ============================================================
# MAIN LOOP
# ============================================================

for cif_file in cif_files:
    name = os.path.splitext(os.path.basename(cif_file))[0]
    atoms = read(cif_file)

    # exactly like relaxation
    atoms.set_pbc([True, True, False])
    natoms = len(atoms)

    start_time = time.time()

    for T in TEMPERATURES:

        # define hard walls from initial geometry
        z = atoms.positions[:, 2]
        z_lo = z.min() - VACUUM_PADDING
        z_hi = z.max() + VACUUM_PADDING

        atoms.calc = ZWallCalculator(
            base_calc=BASE_CALC,
            z_lo=z_lo,
            z_hi=z_hi,
            k_wall=K_WALL,
        )

        # ----------------------------
        # VELOCITIES (CRITICAL PART)
        # ----------------------------
        MaxwellBoltzmannDistribution(atoms, T * units.kB)
        Stationary(atoms)       # remove COM drift
        ZeroRotation(atoms)

        # remove initial z-velocity only
        vel = atoms.get_velocities()
        vel[:, 2] = 0.0
        atoms.set_velocities(vel)

        # ----------------------------
        # OUTPUTS
        # ----------------------------
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

        def safe_step():
            epot = atoms.get_potential_energy()
            if not np.isfinite(epot) or epot > 0.0:
                raise RuntimeError(f"Unphysical energy: {epot:.6e}")

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

        dyn.attach(safe_step, interval=1)

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