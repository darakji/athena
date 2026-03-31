# Li | LLZO Interface ML Workflow

This repository contains the complete computational workflow for building, simulating, and iteratively learning interatomic potentials for **Lithium metal | LLZO solid electrolyte** interfaces — a central system in solid-state battery (SSB) research.

**Key capabilities:**
- End-to-end pipeline from slab generation → DFT labeling → ML training → active learning
- Latent-space-guided structure selection (FPS on MACE embeddings) to minimize DFT calls
- Fine-tuned MACE models via LoRA, avoiding catastrophic forgetting of universal potential knowledge
- Molecular dynamics sampling at elevated temperatures on ALCF Polaris (ML) + OLCF Frontier (DFT)
- Handles ~10³–2,500-atom interface systems across multiple Li and LLZO facets and terminations

---

## Workflow Overview

```
Li/LLZO Slab Generation
       ↓
Lattice Matching & Supercell Construction
       ↓
Decoupled Slab Relaxation (MACE universal model)
       ↓
Gap Energy Scan → Optimal Interface Stacking
       ↓
Full Interface Relaxation
       ↓
MD Sampling at Elevated Temperatures (Polaris, multi-GPU)
       ↓
Latent Embedding Extraction (MACE GNN encoder)
       ↓
Farthest Point Sampling (FPS) → DFT Candidate Selection
       ↓
DFT Labeling via DFT-FE (Frontier, multi-node)
       ↓
LoRA Fine-tuning of MACE on DFT-labeled Structures
       ↓
(Repeat from MD Sampling with updated model)
```

---

## 🏗️ 1. Geometry Generation & Lattice Matching

### Li Slab Generation
* **Script**: `scripts/generate_li_slabs.py`
* **Purpose**: Generates pristine BCC Lithium metal slabs along key cleavage planes `(100)`, `(110)`, and `(111)`. Baseline slabs are constructed using ASE.
* **Output**: Pristine Li `.cif` configurations → `li_slabs/`

### Lattice Matching & Supercell Construction
* **Script**: `scripts/lattice_matching_sc.py`
* **Purpose**: Screens permutations of Li slabs against LLZO slabs. Tests planar expansions up to `5×5×1` to identify minimal interfacial strain pairs. Strains the softer Li lattice to conform with the stiffer LLZO structure while maintaining periodicity.
* **Output**: Decoupled, strain-matched `.cif` structures → `li_and_llzo_unrelaxed_seperate/`

---

## 📉 2. Decoupled Slab Relaxations

### Initial Relaxation
* **Script**: `scripts/relaxation_decoupled.py`
* **Purpose**: Geometry optimization of individual slabs prior to interface construction, using the MACE universal model.
* **Constraints**:
  * **Li**: Top layers frozen to act as a bulk reservoir.
  * **LLZO**: Bottom layers frozen to preserve bulk rigidity.
* **Optimizer**: FIRE (`fmax = 0.05 eV/Å`)
* **Output**: `li_and_llzo_relaxed_seperate/`

---

## 🧲 3. Interface Construction & Energy Scans

### Gap Energy Scan
* **Script**: `scripts/gap_energy_stacking.py`
* **Purpose**: Mechanically slides the Li slab toward the LLZO surface across a range of interfacial gaps (1.0–4.0 Å). Evaluates single-point energies (SPE) at each distance using MACE.
* **Output**: Unrelaxed stacked conformations → `li_llzo_unrelaxed_stacking/`; Markdown log with per-configuration energies.

### Optimal Configuration Selection
* **Script**: `scripts/choose_best_gap.py`
* **Purpose**: Parses gap energy sweep results to automatically select the energetically most favorable interfacial distance for each interface pair.

---

## ⚖️ 4. Global Interface Relaxation

### Full Relaxation
* **Script**: `scripts/relax_slabs.py`
* **Purpose**: Joint structure-wide relaxation of the best-stacked interface using MACE + FIRE optimizer.
* **Output**: Fully relaxed interface structures → `li_llzo_relaxed_bestgaps/`

---

## 🌬️ 5. Molecular Dynamics (MD) Sampling

* **Scripts**: `polaris/scripts/run_md_unfreeze_li.py`, `polaris/scripts/run_md_node0.sh`
* **Purpose**: Time-evolves the interface system to sample thermally accessible configurations unreachable via static optimization.
  * Langevin thermostat MD at multiple elevated temperatures (e.g., 550 K, 1100 K) to accelerate phase space exploration.
  * Geometric constraint masking: deep bulk regions frozen; Li and near-interface LLZO layers fully mobile.
* **Environment**: Distributed multi-GPU execution on ALCF Polaris.

---

## 🧬 6. Latent Embedding Extraction & Downselection

MD trajectories produce large volumes of correlated structural snapshots. A compact, high-variance subset is selected for DFT labeling by embedding frames in the MACE latent space.

### Latent Space Evaluation
* **Scripts**: `polaris/scripts/latent_extraction.sh`, `scripts/extract_embeddings.py`
* **Purpose**: Passes each structural snapshot through the pre-trained MACE GNN encoder. Mean-pools node-level embeddings to produce fixed-size structure-level representations.

### Aggregation & Validation
* **Scripts**: `structure_level_latents/concat_latents.py`, `compare_fps_seeds.py`
* **Purpose**: Concatenates embeddings across compute pools. Validates structural invariance under affine modifications (boundary centering, vacuum resizing).

### Farthest Point Sampling (FPS)
* **Scripts**: `structure_level_latents/fps_maceomat_ver.ipynb`
* **Purpose**: Greedy FPS over the full embedding distribution using Euclidean or cosine distance metrics. Iteratively selects structures that maximize latent space coverage relative to the existing training seed — prioritizing OOD regions and underrepresented interface configurations.
* **Post-processing**: `structure_level_latents/fix_slab.py` — unwraps and centers slabs to remove boundary-wrap artifacts before DFT submission.
* **Output**: Selected candidates → `selected_iteration1_superSeed/`, `selected_iteration1_superSeed_centered/`

---

## ⚛️ 7. DFT Labeling (DFT-FE)

* **Environment**: OLCF Frontier (multi-node, multi-GPU)
* **Purpose**: High-fidelity all-electron DFT energy/force/stress labeling of FPS-selected structures using DFT-FE.
* **Systems**: ~10³–2,500 atoms per structure; ~5,000–15,000 electrons.
* **Output**: Labeled `.extxyz` datasets → `DFTFE_labelled_data/`, DFT run logs → `dftfe_logs/`, `selected_iteration1_superSeed_dftfe_inputs_dftfelogs/`

---

## 🤖 8. Model Training & Fine-Tuning (LoRA)

* **Notebooks/scripts**: `training_npotebook.ipynb`, `mace_fps_training/`, `training_it1/`, `training_it1woburov/`
* **Method**: LoRA (Low-Rank Adaptation) fine-tuning of a universal MACE foundation model on the DFT-labeled interface dataset.
  * LoRA constrains parameter updates to low-rank subspaces, preserving the model's prior knowledge of bulk chemistry while adapting to interface-specific configurations.
  * Evaluated on held-out sets; trained models committed as `mace_seed_*.model` and `mace_seed_*_rank8.model`.
* **Compiled models**: `*_compiled.model` variants for inference-time deployment.

---

## 📊 9. Results & Analysis

* **Directories**: `results/`, `universal_model_results/`, `split1_trial0_modelAndAnalysis/`
* **Notebooks**: `bulk_energy_check.ipynb`, `surface_energy.ipynb`, `NEB_test.ipynb`
* NEB (Nudged Elastic Band) calculations for interface transition state analysis: `neb.traj`, `neb_band.xyz`
* Burov et al. reference structures used for external validation: `Rajdeep_final_structures_burov/`, `Case3VASP/`

---

## 🗂️ Repository Structure

| Path | Contents |
|---|---|
| `scripts/` | Slab generation, relaxation, lattice matching, gap scan |
| `polaris/` | HPC job scripts for MD and embedding extraction (Polaris) |
| `structure_level_latents/` | FPS notebooks, embedding aggregation, seed comparison |
| `mace_fps_training/` | Training data preparation and model fine-tuning |
| `training_it1/`, `training_it1woburov/` | Iteration-1 training runs (with/without Burov reference data) |
| `DFTFE_labelled_data/` | DFT-FE labeled structures (subset) |
| `dftfe_logs/` | DFT-FE run logs |
| `selected_iteration1_superSeed*/` | FPS-selected structures and DFT inputs for iteration 1 |
| `li_and_llzo_*`, `li_llzo_*` | Intermediate geometry files at each pipeline stage |
| `results/`, `universal_model_results/` | Model evaluation and benchmark outputs |
| `checkpoints/` | Training checkpoints |
| `single_atoms/` | Single-atom reference energies for cohesive energy computation |

---

## 📦 Data & Model Access

Due to size constraints, full datasets (`.extxyz`), trained models, and complete DFT logs are not included in this repository.

For access, please contact:
- **mehuldarak@iisc.ac.in**
- **phanim@iisc.ac.in**

---

## Execution Flow Summary

`Li/LLZO Surface Gen` → `Supercell Matching & Strain` → `Decoupled Relaxation` → `Gap Energy Scan` → `Global Interface Relaxation` → `High-Temp MD (Polaris)` → `MACE Embedding Extraction` → `FPS Downselection` → `DFT-FE Labeling (Frontier)` → `LoRA Fine-tuning` → *(iterate)*