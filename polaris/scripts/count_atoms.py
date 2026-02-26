# count_atoms.py
import glob, os
from ase.io import read

in_dir = "/home/phanim/ml/athena/polaris/li_llzo_relaxed_bestgaps_polaris/cifs"

for f in sorted(glob.glob(os.path.join(in_dir, "*.cif"))):
    atoms = read(f)
    print(f"{os.path.basename(f)},{len(atoms)}")