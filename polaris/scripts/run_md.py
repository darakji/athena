# run_md.py
import os, sys, glob, time
import numpy as np

from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import MACECalculator


# ============================================================
# HARD WALL CALCULATOR
# ============================================================

class ZWallCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, z_lo, z_hi, k_wall):
        super().__init__()
        self.base = base_calc
        self.z_lo = z_lo
        self.z_hi = z_hi
        self.k = k_wall

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        self.base.calculate(atoms, properties, system_changes)

        forces = self.base.results["forces"].copy()
        energy = self.base.results.get("energy", 0.0)

        z = atoms.positions[:, 2]

        mask = z < self.z_lo
        dz = self.z_lo - z[mask]
        forces[mask, 2] += self.k * dz
        energy += 0.5 * self.k * np.sum(dz**2)

        mask = z > self.z_hi
        dz = z[mask] - self.z_hi
        forces[mask, 2] -= self.k * dz
        energy += 0.5 * self.k * np.sum(dz**2)

        self.results["forces"] = forces
        self.results["energy"] = energy


# ============================================================
# PATHS
# ============================================================

in_dir = "/home/phanim/ml/athena/polaris/li_llzo_relaxed_bestgaps_polaris/cifs"
out_base = "/home/phanim/ml/athena/polaris/li_llzo_md_polaris"

cif_out  = os.path.join(out_base, "cifs")
traj_out = os.path.join(out_base, "traj")
log_out  = os.path.join(out_base, "logs")

os.makedirs(cif_out, exist_ok=True)
os.makedirs(traj_out, exist_ok=True)
os.makedirs(log_out, exist_ok=True)


# ============================================================
# CALCULATOR
# ============================================================

model_path = "/home/phanim/ml/MACE_models/2024-01-07-mace-128-L2_epoch-199.model"

base_calc = MACECalculator(
    model_paths=model_path,
    device="cuda",
    default_dtype="float32",
)


# ============================================================
# MD PARAMETERS
# ============================================================

temperatures = [300, 600]
timestep_fs = 1.0
nsteps_md = 5000
snapshot_interval = 25

friction = 0.03 / units.fs
vacuum_padding = 8.0
k_wall = 2000.0


# ============================================================
# INPUT CIF SELECTION
# ============================================================

if len(sys.argv) > 1:
    cif_files = [os.path.join(in_dir, f) for f in sys.argv[1:]]
else:
    cif_files = sorted(glob.glob(os.path.join(in_dir, "*.cif")))

print(f"Found {len(cif_files)} structures", flush=True)


# ============================================================
# MAIN LOOP
# ============================================================

for cif_file in cif_files:
    name = os.path.splitext(os.path.basename(cif_file))[0]

    atoms = read(cif_file)
    atoms.set_pbc([True, True, False])
    natoms = len(atoms)

    start_time = time.time()

    for T in temperatures:
        z = atoms.positions[:, 2]
        z_lo = z.min() - vacuum_padding
        z_hi = z.max() + vacuum_padding

        atoms.calc = ZWallCalculator(
            base_calc=base_calc,
            z_lo=z_lo,
            z_hi=z_hi,
            k_wall=k_wall,
        )

        MaxwellBoltzmannDistribution(atoms, T * units.kB)
        ZeroRotation(atoms)

        traj = Trajectory(
            os.path.join(traj_out, f"{name}_T{T}K.traj"),
            "w",
            atoms,
        )

        dyn = Langevin(
            atoms,
            timestep_fs * units.fs,
            T * units.kB,
            friction,
            logfile=os.path.join(log_out, f"{name}_T{T}K.log"),
        )

        step = {"i": 0}

        def safe_step():
            epot = atoms.get_potential_energy()
            if not np.isfinite(epot) or epot > 0.0:
                raise RuntimeError(f"Unphysical energy: {epot:.6e}")
            traj.write(atoms)
            step["i"] += 1
            if step["i"] % snapshot_interval == 0:
                write(
                    os.path.join(cif_out, f"{name}_T{T}K_{step['i']}.cif"),
                    atoms,
                )

        dyn.attach(safe_step, interval=1)

        try:
            dyn.run(nsteps_md)
        except Exception as e:
            print(f"[SKIP] {name} @ {T} K → {e}", flush=True)
        finally:
            traj.close()

    elapsed = time.time() - start_time
    with open(os.path.join(out_base, "MD_TIME_ESTIMATE.md"), "a") as f:
        f.write(
            f"- {name}: {natoms} atoms | "
            f"{elapsed/60:.2f} min | "
            f"{elapsed/natoms:.4f} s/atom\n"
        )

print("ALL MD FINISHED")