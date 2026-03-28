"""
pool_latents.py
---------------
Read one or more .extxyz files that already contain per-atom `mace_latent`
arrays (written by extract_embeddings.py) and produce a structure-level
JSON where each entry maps a unique structure key -> mean-pooled latent.

Key naming conventions (configurable via --key_mode):
  - "filename"    : just the stem of the source file  (default for single-file mode)
  - "source_file" : atoms.info["source_file"] if present, else filename stem
  - "index"       : global integer index across all structures

Usage (single file):
    python pool_latents.py \
        --input /path/to/embeddings.extxyz \
        --output structure_latents.json

Usage (directory of .extxyz):
    python pool_latents.py \
        --input /path/to/embeddings_dir/ \
        --output structure_latents.json \
        --key_mode source_file
"""

import argparse
import json
import os
from pathlib import Path

import ase.io
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Mean-pool per-atom mace_latent -> structure-level JSON"
    )
    p.add_argument(
        "--input", required=True,
        help="Path to a single .extxyz file OR a directory containing multiple .extxyz files"
    )
    p.add_argument(
        "--output", required=True,
        help="Output JSON file path"
    )
    p.add_argument(
        "--array_name", default="mace_latent",
        help="Name of the per-atom array to pool (default: mace_latent)"
    )
    p.add_argument(
        "--key_mode", default="auto",
        choices=["auto", "source_file", "filename", "index"],
        help=(
            "How to key each structure in the output JSON.\n"
            "  auto        : use source_file if present, else filename stem\n"
            "  source_file : atoms.info['source_file'] (raises if missing)\n"
            "  filename    : stem of the .extxyz file being read\n"
            "  index       : global integer index (0-based)"
        )
    )
    p.add_argument(
        "--pool", default="mean",
        choices=["mean", "sum", "max"],
        help="Pooling operation over atoms (default: mean)"
    )
    return p.parse_args()


POOL_FNS = {
    "mean": np.mean,
    "sum":  np.sum,
    "max":  np.max,
}


def collect_extxyz_files(input_path: Path):
    """Return list of .extxyz files to process."""
    if input_path.is_dir():
        files = sorted(input_path.glob("*.extxyz"))
        if not files:
            raise FileNotFoundError(f"No .extxyz files found in {input_path}")
        print(f"Found {len(files)} .extxyz file(s) in directory: {input_path}")
        return files
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return [input_path]


def make_key(atoms, key_mode, file_path: Path, global_idx: int) -> str:
    if key_mode == "index":
        return str(global_idx)
    if key_mode == "source_file":
        src = atoms.info.get("source_file")
        if src is None:
            raise KeyError(
                f"key_mode='source_file' requested but atoms at index {global_idx} "
                f"has no 'source_file' in info dict."
            )
        return src
    if key_mode == "filename":
        return file_path.stem
    # auto
    src = atoms.info.get("source_file")
    if src is not None:
        return src
    return file_path.stem


def pool_array(arr: np.ndarray, pool_fn) -> np.ndarray:
    """arr shape: (n_atoms, latent_dim)  ->  (latent_dim,)"""
    return pool_fn(arr, axis=0)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pool_fn = POOL_FNS[args.pool]
    extxyz_files = collect_extxyz_files(input_path)

    results = {}          # key -> latent list
    global_idx = 0
    skipped = 0

    for fpath in extxyz_files:
        print(f"Processing: {fpath.name}", flush=True)
        try:
            all_atoms = list(ase.io.iread(str(fpath), index=":"))
        except Exception as e:
            print(f"  WARNING: failed to read {fpath.name}: {e}")
            continue

        for atoms in all_atoms:
            if args.array_name not in atoms.arrays:
                print(
                    f"  WARNING: structure #{global_idx} in {fpath.name} "
                    f"has no '{args.array_name}' array — skipping"
                )
                skipped += 1
                global_idx += 1
                continue

            latent_atoms = atoms.arrays[args.array_name]   # (n_atoms, D)
            if latent_atoms.ndim == 1:
                # single atom edge case
                latent_atoms = latent_atoms[np.newaxis, :]

            structure_latent = pool_array(latent_atoms, pool_fn)   # (D,)

            key = make_key(atoms, args.key_mode, fpath, global_idx)

            # Deduplicate keys gracefully
            if key in results:
                key = f"{key}__idx{global_idx}"

            results[key] = structure_latent.tolist()
            global_idx += 1

        print(f"  -> {len(all_atoms)} structures read", flush=True)

    print(f"\nTotal structures pooled : {global_idx - skipped}")
    print(f"Skipped (no array)      : {skipped}")
    print(f"Writing JSON to         : {output_path}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
