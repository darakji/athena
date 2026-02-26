"""
This script generates Li slabs with 3 different Miller indices (100, 110, 111)
and saves them in directory 'li_slabs_fixed_heavy'.
"""
from ase.build import bulk, surface
from ase.io import write
import os

# Output directory
os.makedirs("li_slabs_feb26", exist_ok=True)

# Define bulk Li (bcc)
li_bulk = bulk("Li", "bcc", a=3.51)

# Parameters
miller_indices = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers = 10
vacuum = 15.0
repeat_xy = (1, 1)  # Larger footprint

# Generate and save
for hkl in miller_indices:
    slab = surface(li_bulk, hkl, layers=layers, vacuum=vacuum)
    slab *= (*repeat_xy, 1)
    slab.center(axis=2)
    fname = f"li_slabs_feb26/Li_{hkl[0]}{hkl[1]}{hkl[2]}_slab.cif"
    write(fname, slab)
    print(f"✅ Saved: {fname}")