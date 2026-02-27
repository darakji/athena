from ase.io import read
import numpy as np

traj = read("/home/mehuldarak/athena/polaris/scripts/remaining_slab_md_unfreeze_li/traj/Li_100_slab__LLZO_001_Zr_code93_sto_bestgap_3.00A_r_T400K.traj", index=":")
li_indices = [i for i, s in enumerate(traj[0].get_chemical_symbols()) if s == "Li"]

pos0 = traj[0].positions[li_indices]
msds = []
for atoms in traj:
    disp = atoms.positions[li_indices] - pos0
    msds.append(np.mean(np.sum(disp**2, axis=1)))

import matplotlib.pyplot as plt
plt.plot(msds)
plt.xlabel("Step")
plt.ylabel("MSD (Å²)")
plt.show()