import os
import pandas as pd
from ase.io import iread

# Configuration
SEED_DIR = "/home/mehuldarak/athena/structure_level_latents/fps_seed"
EXTXYZ_PATH = "/home/mehuldarak/athena_embeddings_309machine/309_again_trial0.extxyz"
OUTPUT_EXCEL = "cohesive_energies_fps_seed_309.xlsx"

def run_extraction():
    # 1. Get list of seed filenames (basenames)
    seed_files = set(os.listdir(SEED_DIR))
    print(f"Targeting {len(seed_files)} seed structures from {SEED_DIR}")

    results = []
    processed_count = 0
    match_count = 0

    # 2. Iterate through extxyz
    print(f"Scanning {EXTXYZ_PATH}...")
    for atoms in iread(EXTXYZ_PATH, index=":"):
        processed_count += 1
        
        # Get filename from source_file info
        source_path = atoms.info.get("source_file", "")
        basename = os.path.basename(source_path)

        if basename in seed_files:
            match_count += 1
            energy = atoms.info.get("mace_energy")
            cohesive_energy = atoms.info.get("mace_cohesive_energy")
            
            results.append({
                "Filename": basename,
                "MACE_Energy_eV": energy,
                "MACE_Cohesive_Energy_eV": cohesive_energy,
                "Num_Atoms": len(atoms)
            })
            
            if match_count == len(seed_files):
                print("All seed files found!")
                break

    print(f"Finished. Scanned {processed_count} frames, found {match_count} matches.")

    # 3. Save to Excel
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Results saved to {OUTPUT_EXCEL}")

if __name__ == "__main__":
    run_extraction()
