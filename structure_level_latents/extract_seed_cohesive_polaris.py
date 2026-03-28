import os
import pandas as pd
from ase.io import iread

# Configuration
SEED_DIR = "/home/mehuldarak/athena/structure_level_latents/fps_seed"
# Per-GPU shard files on Polaris
EMBEDDINGS_DIR = "/eagle/DFTCalculations/mehul/ml/athena/polaris/remaining_slab_md_unfreeze_li/embeddings_trial0"
EXTXYZ_FILES = [
    os.path.join(EMBEDDINGS_DIR, f"polaris_emb_trial0_{i}.extxyz")
    for i in range(4)
]
OUTPUT_EXCEL = "cohesive_energies_fps_seed_polaris.xlsx"

def run_extraction():
    # 1. Get list of seed filenames (basenames)
    if not os.path.exists(SEED_DIR):
        print(f"Warning: SEED_DIR {SEED_DIR} not found. Ensure the path is correct on the target machine.")
        return

    seed_files = set(os.listdir(SEED_DIR))
    print(f"Targeting {len(seed_files)} seed structures from {SEED_DIR}")

    results = []
    seen_filenames = set()  # avoid duplicates across shards
    processed_count = 0
    match_count = 0

    # 2. Iterate over all 4 shard files
    for extxyz_path in EXTXYZ_FILES:
        if not os.path.exists(extxyz_path):
            print(f"Warning: {extxyz_path} not found, skipping.")
            continue

        print(f"Scanning {extxyz_path}...")
        try:
            for atoms in iread(extxyz_path, index=":"):
                processed_count += 1

                source_path = atoms.info.get("source_file", "")
                basename = os.path.basename(source_path)

                if basename in seed_files and basename not in seen_filenames:
                    seen_filenames.add(basename)
                    match_count += 1
                    energy = atoms.info.get("mace_energy")
                    cohesive_energy = atoms.info.get("mace_cohesive_energy")

                    results.append({
                        "Filename": basename,
                        "MACE_Energy_eV": energy,
                        "MACE_Cohesive_Energy_eV": cohesive_energy,
                        "Num_Atoms": len(atoms),
                        "Shard": os.path.basename(extxyz_path),
                    })

        except Exception as e:
            print(f"Error reading {extxyz_path}: {e}")

    print(f"Finished. Scanned {processed_count} frames total, found {match_count} matches.")

    # 3. Save to Excel
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Results saved to {OUTPUT_EXCEL}")

if __name__ == "__main__":
    run_extraction()
