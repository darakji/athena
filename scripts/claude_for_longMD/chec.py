from ase.io import read
import numpy as np

# ---- load your structure ----
atoms = read("/home/mehuldarak/athena/li_and_llzo_stacked_for_long_MD_claude/Li_110_slab__longMDstack_v2__LLZO_110_Zr.cif")  # change this

# ---- compute distances ----
d = atoms.get_all_distances(mic=True)

# ignore self-distances
np.fill_diagonal(d, 1e6)

min_dist = d.min()
max_dist = d.max()

print("="*50)
print(f"Number of atoms: {len(atoms)}")
print(f"Minimum distance: {min_dist:.4f} Å")
print(f"Maximum distance: {max_dist:.4f} Å")

# ---- find problematic pairs ----
threshold = 1.5  # Å (you can adjust)

print("\nPairs closer than threshold:")
indices = np.where(d < threshold)

for i, j in zip(*indices):
    if i < j:
        print(f"Atoms {i}-{j}: {d[i,j]:.4f} Å | {atoms[i].symbol}-{atoms[j].symbol}")

print("="*50)