import os
import numpy as np
from itertools import product
from ase.io import read, write
from ase.geometry import cell_to_cellpar
from ase.build import rotate

# ==================================================
# Paths
# ==================================================
li_dir = "/home/mehuldarak/athena/li_slabs"
llzo_dir = "/home/mehuldarak/athena/llzo_slabs"

out_base = "/home/mehuldarak/athena/li_and_llzo_unrelaxed_seperate_2500_3000"
log_file = "/home/mehuldarak/athena/li_and_llzo_unrelaxed_seperate_2500_3000/li_llzo_lattice_matching_report_2500_3000.md"

os.makedirs(out_base, exist_ok=True)

# ==================================================
# Params
# ==================================================
MAX_REPEAT = 5
MAX_ALLOWED_MISMATCH = 0.10
TARGET_MIN_ATOMS = 2000
TARGET_MAX_ATOMS = 3000
BUFFER = 2.0


# ==================================================
# 🔥 CRITICAL: Align slab normal to z
# ==================================================
def align_slab_to_z(atoms):
    atoms = atoms.copy()

    cell = atoms.cell.array

    # pick longest lattice vector → likely vacuum direction
    lengths = [np.linalg.norm(v) for v in cell]
    normal_idx = np.argmax(lengths)

    normal = cell[normal_idx]

    # normalize
    normal = normal / np.linalg.norm(normal)

    # rotate normal → z
    z = np.array([0, 0, 1.0])

    # cross + angle
    axis = np.cross(normal, z)
    angle = np.arccos(np.clip(np.dot(normal, z), -1, 1)) * 180 / np.pi

    if np.linalg.norm(axis) > 1e-6:
        rotate(atoms, angle, axis, rotate_cell=True)

    return atoms


# ==================================================
# Remove vacuum AFTER alignment
# ==================================================
def make_dense(atoms):
    pos = atoms.positions

    zmin = pos[:, 2].min()
    zmax = pos[:, 2].max()

    thickness = zmax - zmin

    pos[:, 2] -= zmin
    atoms.positions = pos

    a = atoms.cell[0]
    b = atoms.cell[1]

    atoms.set_cell([a, b, [0, 0, thickness + BUFFER]], scale_atoms=False)
    atoms.wrap()

    return atoms


def compact_z(atoms):
    pos = atoms.positions
    zmin = pos[:, 2].min()
    zmax = pos[:, 2].max()

    thickness = zmax - zmin

    pos[:, 2] -= zmin
    atoms.positions = pos

    cell = atoms.cell.copy()
    cell[2] = [0, 0, thickness + BUFFER]

    atoms.set_cell(cell, scale_atoms=False)
    atoms.wrap()

    return atoms


# ==================================================
# Files
# ==================================================
li_files = sorted(f for f in os.listdir(li_dir) if f.endswith(".cif"))
llzo_files = sorted(f for f in os.listdir(llzo_dir) if f.endswith(".cif"))

# ==================================================
# Log
# ==================================================
with open(log_file, "w") as f:
    f.write("# Li–LLZO Lattice Matching Report (≤10%)\n\n")


# ==================================================
# Main
# ==================================================
for li_name, llzo_name in product(li_files, llzo_files):

    li = read(os.path.join(li_dir, li_name))
    llzo = read(os.path.join(llzo_dir, llzo_name))

    # 🔥 ALIGN FIRST (THIS WAS MISSING)
    li = align_slab_to_z(li)
    llzo = align_slab_to_z(llzo)

    # remove vacuum
    li = make_dense(li)
    llzo = make_dense(llzo)

    best = None

    for li_i, li_j, llzo_i, llzo_j in product(
        range(1, MAX_REPEAT + 1),
        range(1, MAX_REPEAT + 1),
        range(1, MAX_REPEAT + 1),
        range(1, MAX_REPEAT + 1),
    ):

        li_tmp = li.repeat((li_i, li_j, 1))
        llzo_tmp = llzo.repeat((llzo_i, llzo_j, 1))

        a_li = np.linalg.norm(li_tmp.cell[0])
        b_li = np.linalg.norm(li_tmp.cell[1])
        a_llzo = np.linalg.norm(llzo_tmp.cell[0])
        b_llzo = np.linalg.norm(llzo_tmp.cell[1])

        mismatch_a = abs(a_li - a_llzo) / max(a_li, a_llzo)
        mismatch_b = abs(b_li - b_llzo) / max(b_li, b_llzo)

        if mismatch_a > MAX_ALLOWED_MISMATCH or mismatch_b > MAX_ALLOWED_MISMATCH:
            continue

        score = max(mismatch_a, mismatch_b)

        base_atoms = len(li_tmp) + len(llzo_tmp)

        z_repeat = max(1, TARGET_MIN_ATOMS // base_atoms)

        total_atoms = base_atoms * z_repeat

        if total_atoms > TARGET_MAX_ATOMS:
            continue

        if best is None or score < best[0]:
            best = (
                score,
                (li_i, li_j),
                (llzo_i, llzo_j),
                a_li,
                b_li,
                a_llzo,
                b_llzo,
                mismatch_a,
                mismatch_b,
                z_repeat,
                total_atoms
            )

    if best is None:
        print(f"[FILTERED] {li_name} + {llzo_name}")
        continue

    _, li_rep, llzo_rep, a_li, b_li, a_llzo, b_llzo, ma, mb, zrep, tot = best

    li_sc = li.repeat((li_rep[0], li_rep[1], 1))
    llzo_sc = llzo.repeat((llzo_rep[0], llzo_rep[1], 1))

    # scale Li
    scale_a = a_llzo / a_li
    scale_b = b_llzo / b_li

    cell = li_sc.cell.copy()
    cell[0] *= scale_a
    cell[1] *= scale_b

    li_sc.set_cell(cell, scale_atoms=True)

    # match cell
    li_sc.set_cell(llzo_sc.cell, scale_atoms=False)

    # repeat z
    li_sc = li_sc.repeat((1, 1, zrep))
    llzo_sc = llzo_sc.repeat((1, 1, zrep))

    # 🔥 FINAL compact
    li_sc = compact_z(li_sc)
    llzo_sc = compact_z(llzo_sc)

    # save
    li_tag = os.path.splitext(li_name)[0]
    llzo_tag = os.path.splitext(llzo_name)[0]

    folder = os.path.join(out_base, f"{li_tag}__{llzo_tag}")
    os.makedirs(folder, exist_ok=True)

    write(os.path.join(folder,
        f"{li_tag}__sc_{li_rep[0]}x{li_rep[1]}x1__scaled_to_{llzo_tag}.cif"), li_sc)

    write(os.path.join(folder,
        f"{llzo_tag}__sc_{llzo_rep[0]}x{llzo_rep[1]}x1__fixed.cif"), llzo_sc)

    print(f"[OK] {li_name} + {llzo_name}")