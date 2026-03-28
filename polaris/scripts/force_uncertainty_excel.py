"""
Post-processing script: ensemble force uncertainty.

Reads the 4 extxyz output files produced by extract_embeddings.py (one per
MACE model), aligns structures by source_file, computes per-atom and
per-structure force uncertainty, then writes an Excel file ranked by
percentile.

Usage:
  python force_uncertainty_excel.py \
      --embeddings_dir /path/to/embeddings \
      --output_excel   /path/to/output.xlsx
"""

import argparse
import glob
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import ase.io
import pandas as pd


# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings_dir", required=True,
                    help="Directory containing one extxyz per model (ensemble_*.extxyz)")
parser.add_argument("--output_excel", required=True,
                    help="Output Excel file path")
args = parser.parse_args()

embeddings_dir = Path(args.embeddings_dir)
output_excel   = Path(args.output_excel)
output_excel.parent.mkdir(parents=True, exist_ok=True)

# ── Load all extxyz files ─────────────────────────────────────────────────────

extxyz_files = sorted(embeddings_dir.glob("ensemble_*.extxyz"))
if not extxyz_files:
    raise FileNotFoundError(f"No ensemble_*.extxyz files found in {embeddings_dir}")

print(f"Found {len(extxyz_files)} model output files:")
for f in extxyz_files:
    print(f"  {f.name}")

# model_tag → {source_file → atoms}
model_data: dict[str, dict[str, object]] = {}

for f in extxyz_files:
    tag = f.stem.replace("ensemble_", "")
    structs = {}
    for atoms in ase.io.iread(f, index=":"):
        src = atoms.info.get("source_file", atoms.info.get("filename", str(f)))
        structs[src] = atoms
    model_data[tag] = structs
    print(f"  [{tag}] loaded {len(structs)} structures")

# ── Align structures across models ────────────────────────────────────────────

# Only keep structures present in ALL models
all_keys = [set(v.keys()) for v in model_data.values()]
common_keys = sorted(set.intersection(*all_keys))
print(f"\nStructures present in all {len(model_data)} models: {len(common_keys)}")

if len(common_keys) == 0:
    raise RuntimeError(
        "No structures in common across all model outputs. "
        "Check that source_file keys match across extxyz files."
    )

tags = sorted(model_data.keys())

# ── Compute ensemble force uncertainty ────────────────────────────────────────

rows = []

for src in common_keys:
    struct_name = os.path.basename(src)

    # Stack forces: shape (n_models, n_atoms, 3)
    forces_stack = np.stack(
        [model_data[tag][src].arrays["mace_forces"] for tag in tags],
        axis=0,
    )
    energies = [float(model_data[tag][src].info["mace_energy"]) for tag in tags]

    n_atoms = forces_stack.shape[1]

    # Per-atom force magnitude std across models: shape (n_atoms,)
    # For each atom, compute std of force magnitude (scalar) across models
    force_magnitudes = np.linalg.norm(forces_stack, axis=-1)  # (n_models, n_atoms)
    per_atom_std = force_magnitudes.std(axis=0)                # (n_atoms,)

    # Per-structure summary statistics
    mean_force_std    = float(per_atom_std.mean())
    max_force_std     = float(per_atom_std.max())
    max_force_std_idx = int(per_atom_std.argmax())

    # Also compute std of per-model per-structure max force magnitude
    max_force_per_model = force_magnitudes.max(axis=1)         # (n_models,)
    std_of_max_forces   = float(max_force_per_model.std())

    # Energy spread
    energy_mean = float(np.mean(energies))
    energy_std  = float(np.std(energies))

    # Per-model energies
    model_energy_cols = {f"energy_{tag}": e for tag, e in zip(tags, energies)}

    row = {
        "structure":             struct_name,
        "source_file":           src,
        "n_atoms":               n_atoms,
        "mean_force_std_eV_A":   mean_force_std,
        "max_force_std_eV_A":    max_force_std,
        "max_force_std_atom_idx":max_force_std_idx,
        "std_of_max_forces":     std_of_max_forces,
        "energy_mean_eV":        energy_mean,
        "energy_std_eV":         energy_std,
        **model_energy_cols,
    }
    rows.append(row)

df = pd.DataFrame(rows)

# ── Percentile ranking ────────────────────────────────────────────────────────
# Rank by mean_force_std (primary uncertainty metric), highest = most uncertain

df["percentile_rank"] = df["mean_force_std_eV_A"].rank(pct=True) * 100
df = df.sort_values("percentile_rank", ascending=False).reset_index(drop=True)
df.insert(0, "rank", df.index + 1)

# ── Write Excel ───────────────────────────────────────────────────────────────

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:

    # Sheet 1: Full sorted table
    df.to_excel(writer, sheet_name="Uncertainty_Ranked", index=False)

    # Sheet 2: Top-10% most uncertain
    top10 = df[df["percentile_rank"] >= 90].copy()
    top10.to_excel(writer, sheet_name="Top10pct_Uncertain", index=False)

    # Sheet 3: Per-model summaries
    summary_rows = []
    for tag in tags:
        model_structs = model_data[tag]
        energies_all  = [float(a.info["mace_energy"]) for a in model_structs.values()]
        summary_rows.append({
            "model":              tag,
            "n_structures":       len(model_structs),
            "mean_energy_eV":     float(np.mean(energies_all)),
            "std_energy_eV":      float(np.std(energies_all)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(writer, sheet_name="Model_Summary", index=False)

print(f"\nExcel written to: {output_excel}")
print(f"  Total structures ranked: {len(df)}")
print(f"  Top-10% uncertain (≥90th percentile): {len(top10)}")
print(f"\nTop 5 most uncertain structures:")
print(df[["rank", "structure", "mean_force_std_eV_A", "max_force_std_eV_A",
          "percentile_rank"]].head(5).to_string(index=False))
