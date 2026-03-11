from ase.io import read, write
import glob
import os

# Input directory
cif_dir = "/home/mehuldarak/athena/structure_level_latents/fps_seed_centered"

# Output file
out_file = os.path.join(cif_dir, "fps_seed_centered.extxyz")

# Collect CIFs (sorted for reproducibility)
cif_files = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))

assert len(cif_files) > 0, "No CIF files found!"

frames = []

for cif in cif_files:
    atoms = read(cif)
    atoms.info["source_cif"] = os.path.basename(cif)  # optional metadata
    frames.append(atoms)

# Write all as different frames
write(out_file, frames, format="extxyz")

print(f"Wrote {len(frames)} frames to {out_file}")