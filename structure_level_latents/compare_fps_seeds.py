import os
import json
import subprocess
import numpy as np
import pandas as pd
from ase.io import read
from scipy.spatial.distance import cosine, euclidean
import copy

BASE_DIR = "/home/mehuldarak/athena/structure_level_latents"
CENTERED_DIR = os.path.join(BASE_DIR, "fps_seed_centered")
PREV_JSON = os.path.join(BASE_DIR, "structure_level_latents_all.json")

MODEL_PATH = "/eagle/DFTCalculations/mehul/ml/MACE_models/mace-omat-0-medium.model"
# Fallback to local if polaris path doesn't exist
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/home/mehuldarak/MACE_models/universal_09072025/mace-omat-0-medium.model"
    
EXTRACT_SCRIPT = "/home/mehuldarak/mace/mace/extract_embeddings.py"
OUTPUT_EXTXYZ = os.path.join(BASE_DIR, "fps_seed_centered.extxyz")

print("1. Loading previous latents from JSON...")
with open(PREV_JSON, 'r') as f:
    prev_data = json.load(f)

prev_latents = {}
for entry in prev_data:
    # Handle both full paths and basenames in the JSON
    fname = os.path.basename(entry["file"])
    base_cif = fname if fname.endswith(".cif") else fname + ".cif"
    prev_latents[base_cif] = np.array(entry["structure_embedding"], dtype=np.float32)

print(f"Loaded {len(prev_latents)} entries from previous JSON.")

print("\n2. Extracting latents for centered structures...")
if os.path.exists(OUTPUT_EXTXYZ):
    os.remove(OUTPUT_EXTXYZ)

cmd = [
    "python", EXTRACT_SCRIPT,
    "--model", MODEL_PATH,
    "--configs", CENTERED_DIR,
    "--output", OUTPUT_EXTXYZ,
    "--device", "cuda",
    "--batch_size", "10"
]

print(f"Running: {' '.join(cmd)}")
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode != 0:
    print("Error extracting embeddings:")
    print(res.stderr)
    exit(1)
    
print("Extraction successful.")

print("\n3. Processing new latents and comparing...")
atoms_list = read(OUTPUT_EXTXYZ, index=":")

results = []
for atoms in atoms_list:
    src_file = atoms.info.get("source_file", "unknown")
    base_cif = os.path.basename(src_file)
    
    # Compute structure-level embedding (mean pooling of node feats)
    node_feats = atoms.arrays["mace_latent"].astype(np.float32)
    new_emb = np.mean(node_feats, axis=0)
    
    if base_cif not in prev_latents:
        print(f"WARNING: Configuration {base_cif} not found in previous latents JSON. Skipping.")
        continue
        
    old_emb = prev_latents[base_cif]
    
    # Calculate distances
    l2_dist = euclidean(old_emb, new_emb)
    cos_sim = 1.0 - cosine(old_emb, new_emb)
    rel_change = np.linalg.norm(old_emb - new_emb) / (np.linalg.norm(old_emb) + 1e-12)
    
    results.append({
        "Filename": base_cif,
        "L2_Distance": l2_dist,
        "Cosine_Similarity": cos_sim,
        "Relative_Change": rel_change
    })

df = pd.DataFrame(results)

print("\n" + "="*80)
print("COMPARISON RESULTS (Original vs Centered)")
print("="*80)
print(df.to_string(index=False))

# Calculate summary statistics
print("\nSUMMARY STATISTICS:")
print(f"Mean L2 Distance    : {df['L2_Distance'].mean():.6e}")
print(f"Max L2 Distance     : {df['L2_Distance'].max():.6e}")
print(f"Mean Cosine Sim     : {df['Cosine_Similarity'].mean():.6f}")
print(f"Min Cosine Sim      : {df['Cosine_Similarity'].min():.6f}")
print(f"Mean Relative Change: {df['Relative_Change'].mean():.6e}")

out_csv = os.path.join(BASE_DIR, "centered_comparison_results.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved full results to {out_csv}")
