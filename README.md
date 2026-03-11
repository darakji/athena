# Athena Machine Learning Simulation Workflow

This repository contains the complete computational workflow for generating, combining, simulating, and sampling Solid-State Battery (SSB) interfaces, specifically crossing Lithium (Li) metal anodes and LLZO solid-state electrolytes. 

The primary physical model driving configurations, relaxations, and dynamics is the **MACE (Machine Learning Interatomic Potential)** universal foundational model.

---

## 🏗️ 1. Geometry Generation & Lattice Matching

### Li Slab Generation
* **Script**: `scripts/generate_li_slabs.py`
* **Purpose**: Generates pristine Bcc Lithium metal slabs along key cleavage planes `(100)`, `(110)`, and `(111)`. The baseline slabs are constructed using ASE.
* **Output**: Pristine Li `.cif` configurations.

### Lattice Matching & Supercell Construction
* **Script**: `scripts/lattice_matching_sc.py`
* **Purpose**: Screens permutations of Li slabs against LLZO slabs. Tests planar expansions up to `5x5x1` to identify minimal interfacial strain matching pairs. Strains the softer Li lattice to perfectly conform with the stiffer LLZO structure, maintaining periodicity.
* **Output**: Decoupled, scaled `.cif` structures located in `li_and_llzo_unrelaxed_seperate/`

---

## 📉 2. Decoupled Slab Relaxations

### Initial Relaxation
* **Script**: `scripts/relaxation_decoupled.py`
* **Purpose**: Prior to joining the interfaces, the individual slabs undergo geometry optimization using the MACE calculator. 
* **Constraints**: Implements specific boundary constraints:
  * **Li**: Top layers are frozen to act as a bulk reservoir.
  * **LLZO**: Bottom layers are frozen preserving bulk rigid behavior.
* **Optimiser**: FIRE optimizer (`fmax=0.05`).
* **Output**: Target configurations saved in `li_and_llzo_relaxed_seperate/`.

---

## 🧲 3. Interface Construction & Energy Scans

### Gap Energy Scan
* **Script**: `scripts/gap_energy_stacking.py`
* **Purpose**: Mechanically slides the Li slab toward the LLZO surface across a range of interfacial gaps (e.g., 1.0Å to 4.0Å). Evaluates the Single Point Energy (SPE) for each distance using MACE.
* **Output**: Collection of stacked unrelaxed conformations in `li_llzo_unrelaxed_stacking/` and a comprehensive Markdown log report detailing the energies.

### Optimal Configuration Selection
* **Script**: `scripts/choose_best_gap.py`
* **Purpose**: Automatically parses the gap energy sweep results to select the energetically most favorable distance for every specific interface pair.

---

## ⚖️ 4. Global Interface Relaxation

### Full Relaxation
* **Script**: `scripts/relax_slabs.py`
* **Purpose**: Performs a joint structure-wide relaxation on the best stacked interface. Minimizes the combined Li-LLZO structure with the MACE calculator via the FIRE algorithm. 
* **Output**: Fully minimized structures deposited in `li_llzo_relaxed_bestgaps/`.

---

## 🌬️ 5. Molecular Dynamics (MD) Sampling 

* **Scripts**: `polaris/scripts/run_md_unfreeze_li.py`, shell handlers like `polaris/scripts/run_md_node0.sh`
* **Purpose**: Propagates the system in time to simulate thermal evolution and capture realistic interface rearrangements natively unattainable by static optimizations. 
  * Runs parallel distributed Molecular Dynamics using Langevin thermostats at multiple elevated temperatures (e.g. 550K, 1100K) to accelerate sampling.
  * Uses intelligent geometric constraint masking to preserve deep bulk regions while permitting complete interfacial reconstructions (Li fully mobile, top interface LLZO mobile).
* **Environment**: Executed concurrently across heterogeneous HPC clusters or GPU nodes (e.g., ALCF Polaris).

---

## 🧬 6. Latent Embedding Extraction & Downselection

The MD simulations create a massive volume of correlated structural snapshots. To select a compact, high-variance dataset for high-fidelity ab-initio (DFT) investigation, we embed frames in the MACE latent space.

### Latent Space Evaluation
* **Scripts**: `polaris/scripts/latent_extraction.sh`, `scripts/extract_embeddings.py`
* **Purpose**: Evaluates pre-trained MACE Graph Neural Network encodings for each structure snapshot. Captures rich local and global topological data in `float32` high-dimensional tensors.

### Comparison & Concatenation
* **Scripts**: `structure_level_latents/concat_latents.py`, `compare_fps_seeds.py`
* **Purpose**: Handles aggregation of latents uniformly over multiple compute pools. Also confirms structural invariance under affine modifications (like periodic boundary centering/vacuum resizing).

### Farthest Point Sampling (FPS)
* **Scripts**: Notebooks such as `structure_level_latents/fps_maceomat_ver.ipynb`
* **Purpose**: Conducts Greedy Farthest Point Sampling across the entire distribution using Euclidean or Cosine distance metrics. Iteratively isolates geometrically and chemically diverse snapshots maximizing the phase space coverage of the ultimate dataset.
* **Refinement**: `structure_level_latents/fix_slab.py` is invoked to unwrap and artificially center the active slabs, averting boundary-wrap visualization artifacts and ensuring cleaner configurations.

---

### Project Execution Flow Summary
`Lithium/LLZO Surface Gen` ➡️ `Supercell Match/Strain` ➡️ `Decoupled Relaxation` ➡️ `Gap Scan Stacking` ➡️ `Global Structure Minimization` ➡️ `High-Temp Molecular Dynamics` ➡️ `MACE Embeddings Extraction` ➡️ `Latent Phase Farthest Point Sampling (FPS)` ➡️ `Subsequent High-Fidelity Calculations`.
