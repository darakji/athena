from ase.io import read
from mace.calculators import MACECalculator
import torch
import numpy as np
import os

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Calculator
# -------------------------
calc = MACECalculator(
    model_paths=[
        "/home/mehuldarak/athena/mace_fps_training/checkpoints/mace_fps_split17_SaveIT0_256_candidate3.model"
    ],
    device=device,
    default_dtype="float32",
    batch_size=16
)

# -------------------------
# Paths
# -------------------------
interface_path = "/home/mehuldarak/athena/li_llzo_best_gap_structures_relaxed_forlongMD/BEST__Li_100_slab__LLZO_001_Zr_code93_sto__gap_2.5A__Eint_-18.33423eVA2__Nvac14_relaxed.cif"

llzo_path = "/home/mehuldarak/athena/archive/bulk_references/Li_100_slab__LLZO_001_Zr_code93_sto/bulk_llzo_vacancy__Li_100_slab__LLZO_001_Zr_code93_sto__nvac14.cif"

li_path = "/home/mehuldarak/athena/bulk_references/Li_100_slab__LLZO_001_Zr_code93_sto/bulk_li__Li_100_slab__LLZO_001_Zr_code93_sto__c1__ip1x5.cif"

single_atom_dir = "/home/mehuldarak/athena/single_atoms"

# -------------------------
# Load structures
# -------------------------
interface = read(interface_path)
llzo = read(llzo_path)
li = read(li_path)
print("Interface cell:\n", interface.get_cell())
print("LLZO cell:\n", llzo.get_cell())
print("Li cell:\n", li.get_cell())
# -------------------------
# Attach calc
# -------------------------
for a in [interface, llzo, li]:
    a.calc = calc

# -------------------------
# Load single atoms
# -------------------------
atomic_energies = {}

for f in os.listdir(single_atom_dir):
    if f.endswith(".cif"):
        atom = read(os.path.join(single_atom_dir, f))
        atom.calc = calc
        symbol = atom.get_chemical_symbols()[0]
        atomic_energies[symbol] = atom.get_potential_energy()

print("Atomic energies:", atomic_energies)

# -------------------------
# Cohesive energy function
# -------------------------
def cohesive_energy(atoms):
    symbols = atoms.get_chemical_symbols()
    E_total = atoms.get_potential_energy()
    E_atoms = sum(atomic_energies[s] for s in symbols)
    return E_total - E_atoms

# -------------------------
# Compute cohesive energies
# -------------------------
E_coh_interface = cohesive_energy(interface)
E_coh_llzo = cohesive_energy(llzo)
E_coh_li = cohesive_energy(li)

# -------------------------
# Area
# -------------------------
cell = interface.get_cell()
A = np.linalg.norm(np.cross(cell[0], cell[2]))

# -------------------------
# Interface energy
# -------------------------
gamma = (E_coh_interface - E_coh_llzo - E_coh_li) / (2 * A)
gamma_J = gamma * 16.0218

print("gamma (eV/Å^2):", gamma)
print("gamma (J/m^2):", gamma_J)