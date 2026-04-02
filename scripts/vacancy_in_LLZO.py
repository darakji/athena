from ase.io import read, write
import numpy as np
import os
import json

# ==================================================
# Path
# ==================================================
IN_BASE = "/home/mehuldarak/athena/li_and_llzo_final_interfaces_2500_3000"

# ==================================================
# Vacancy function
# ==================================================
def create_li_vacancies(llzo, n_remove=2, seed=42):
    np.random.seed(seed)

    symbols = llzo.get_chemical_symbols()

    # Identify Li atoms
    li_indices = [i for i, s in enumerate(symbols) if s == "Li"]

    if len(li_indices) < n_remove:
        raise ValueError("Not enough Li atoms to remove")

    # Select atoms to remove
    remove_indices = np.random.choice(li_indices, size=n_remove, replace=False)
    remove_indices = sorted(remove_indices)

    # Store positions BEFORE removal
    removed_positions = llzo.positions[remove_indices]

    # Remove atoms (reverse order!)
    for idx in sorted(remove_indices, reverse=True):
        del llzo[idx]

    # JSON-safe metadata
    metadata = {
        "removed_indices": [int(i) for i in remove_indices],
        "removed_positions": [[float(x) for x in pos] for pos in removed_positions],
        "n_removed": int(n_remove),
    }

    return llzo, metadata


# ==================================================
# Main loop
# ==================================================
for interface_dir in os.listdir(IN_BASE):

    path = os.path.join(IN_BASE, interface_dir)
    if not os.path.isdir(path):
        continue

    print(f"\nProcessing: {interface_dir}")

    for f in os.listdir(path):

        # ONLY modify LLZO files
        if f.endswith("__fixed.cif"):

            llzo_path = os.path.join(path, f)
            llzo = read(llzo_path)

            # Optional safety check (ensure LLZO-like)
            symbols = llzo.get_chemical_symbols()
            if not {"La", "Zr", "O"}.issubset(set(symbols)):
                print("  ⚠️ Skipping (not LLZO)")
                continue

            # Compute number of Li to remove
            n_atoms = len(llzo)
            n_remove = max(1, int(round(2 * n_atoms / 192)))

            print(f"  Removing {n_remove} Li atoms")

            # Create vacancies
            llzo_new, meta = create_li_vacancies(llzo, n_remove=n_remove)

            # Save vacancy CIF (same directory)
            out_cif = os.path.join(
                path,
                f.replace("__fixed.cif", "__vacancy.cif")
            )
            write(out_cif, llzo_new)

            # Save metadata
            meta_file = os.path.join(path, "vacancy_meta.json")
            with open(meta_file, "w") as fp:
                json.dump(meta, fp, indent=2)

            print(f"  ✔ Saved: {out_cif}")

print("\n✔ Done: LLZO vacancy structures generated in-place")