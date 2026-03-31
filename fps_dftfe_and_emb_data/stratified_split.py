"""
Stratified split of fps_dftfe_data.extxyz into 4 folds.
Each fold: 16 train + 4 val, stratified by number of atoms.

Strategy:
  1. Read all frames and record (index, n_atoms).
  2. Sort frames by n_atoms, then assign a stratum index (0-3) by binning
     into quartiles of n_atoms so every bin has ~5 frames.
  3. Within each stratum, shuffle with a fixed seed and round-robin assign
     frames to the 4 splits (so each split gets ~1 frame per stratum).
  4. For each split, the 4 frames NOT in that split become val; the
     remaining 16 become train.

Total = 20 frames → 4 splits × 4 val = 16 val assignments (each frame
appears as val in exactly 1 split and train in the other 3).
"""

import os
import random
from pathlib import Path

SEED = 42
INPUT = Path("/home/mehuldarak/athena/fps_dftfe_and_emb_data/fps_dftfe_data_it1_full_woburov.extxyz")
OUT_BASE = Path("/home/mehuldarak/athena/splits_it1_woburov")
N_SPLITS = 20
N_VAL = 5  # frames per val set
N_TRAIN = 19  # frames per train set
N_STRATA = N_VAL


# ── 1. Parse all frames ──────────────────────────────────────────────────────

def read_extxyz_frames(path: Path) -> list[list[str]]:
    """Return list of frames; each frame is a list of raw lines."""
    frames = []
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            raise ValueError(f"Expected atom count at line {i+1}, got: {line!r}")
        # frame = atom-count line + comment line + n_atoms data lines
        frame_lines = lines[i : i + n_atoms + 2]
        frames.append(frame_lines)
        i += n_atoms + 2

    return frames


frames = read_extxyz_frames(INPUT)
n_total = len(frames)
print(f"Total frames: {n_total}")
assert n_total == 24, f"Expected 20 frames, got {n_total}"

# ── 2. Extract n_atoms per frame ─────────────────────────────────────────────

n_atoms_list = [int(f[0].strip()) for f in frames]
frame_info = list(enumerate(n_atoms_list))  # [(idx, n_atoms), ...]
print("n_atoms per frame:", [x[1] for x in frame_info])

# ── 3. Stratify by n_atoms → N_STRATA strata ─────────────────────────────────

sorted_by_atoms = sorted(frame_info, key=lambda x: x[1])
stratum_size = n_total // N_STRATA

strata = [sorted_by_atoms[i * stratum_size:(i + 1) * stratum_size]
          for i in range(N_STRATA)]

print("\nStrata (by n_atoms):")
for k, s in enumerate(strata):
    print(f"  Stratum {k}: indices={[x[0] for x in s]}, n_atoms={[x[1] for x in s]}")

# ── 4. Assign frames → splits ────────────────────────────────────────────────
# We assign exactly 1 frame as val to each of the N_SPLITS splits. Since there are
# N_STRATA layers, each split gets exactly N_STRATA val frames. We cycle through
# the stratum to ensure we have enough items if N_SPLITS > stratum_size.

rng = random.Random(SEED)
split_val_indices = [[] for _ in range(N_SPLITS)]  # val frame indices per split

for stratum in strata:
    shuffled = list(stratum)
    rng.shuffle(shuffled)
    
    items = []
    while len(items) < N_SPLITS:
        items.extend(shuffled)
        
    for split_id in range(N_SPLITS):
        frame_idx = items[split_id][0]
        split_val_indices[split_id].append(frame_idx)

print("\nVal indices per split:")
for s, val_idx in enumerate(split_val_indices):
    print(f"  Split {s+1}: val={sorted(val_idx)} (n={len(val_idx)})")

# Verify counts
for s, val_idx in enumerate(split_val_indices):
    assert len(val_idx) == N_VAL, (
        f"Split {s+1} has {len(val_idx)} val frames, expected {N_VAL}"
    )

# ── 5. Write output files ─────────────────────────────────────────────────────

def write_frames(path: Path, frame_list: list[list[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame in frame_list:
            f.writelines(frame)
    print(f"  Wrote {len(frame_list)} frames → {path}")


all_indices = set(range(n_total))

for split_id in range(1, N_SPLITS + 1):
    val_idx = set(split_val_indices[split_id - 1])
    train_idx = sorted(all_indices - val_idx)
    val_idx_sorted = sorted(val_idx)

    split_dir = OUT_BASE / f"split_{split_id}"
    train_path = split_dir / f"fps_dftfe_split_{split_id}_train.extxyz"
    val_path   = split_dir / f"fps_dftfe_split_{split_id}_val.extxyz"

    print(f"\nSplit {split_id}:")
    write_frames(train_path, [frames[i] for i in train_idx])
    write_frames(val_path,   [frames[i] for i in val_idx_sorted])

print("\nDone! All splits written.")
