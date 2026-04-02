from ase.io import Trajectory, write
import os

traj_path = "/home/mehuldarak/clones/relax.traj"
out_path = "/home/mehuldarak/athena/li_llzo_best_gap_structures_relaxed_forlongMD/BEST__Li_100_slab__LLZO_001_Zr_code93_sto__gap_2.5A__Eint_-18.33423eVA2__Nvac14_relaxed.cif"

# create directory if missing
os.makedirs(os.path.dirname(out_path), exist_ok=True)

traj = Trajectory(traj_path)

atoms = traj[485]  # 486th frame

write(out_path, atoms)

print("Saved successfully.")