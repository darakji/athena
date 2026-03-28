"""
Generates fps_coverage_analysis_trial0.ipynb programmatically.
Run:  python3 gen_notebook.py
"""
import json, textwrap, os

NB_PATH = "/home/mehuldarak/athena/structure_level_latents/fps_coverage_analysis_trial0.ipynb"

def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": [src]}

def code(src):
    # split into lines, add \n except last
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}


cells = []

# ── 0 TITLE ──────────────────────────────────────────────────────────────────
cells.append(md("""\
# Active-Learning Iteration 0 — Coverage Analysis (maceTrial0)

**Context**

| Group | N | Meaning |
|---|---|---|
| **T** | 6 | Used to **finetune** MACE → maceTrial0 |
| **V** | 2 | Held-out **validation** during finetuning |
| **Te** | 2 | **Test** set (unseen during training) |
| **FPS_other** | 10 | Other fps_seed candidates (next AL iteration) |
| **pool** | ~5277 | Everything else |

> **Question**: In the *finetuned* model's latent space, how well does T cover V, Te, and the rest of the pool?
"""))

# ── 1 IMPORTS ─────────────────────────────────────────────────────────────────
cells.append(md("## 1. Imports & Config"))
cells.append(code("""\
import json
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
JSON_1       = "/home/mehuldarak/athena/structure_level_latents/maceTrial0_structure_latents.json"
JSON_2       = "/home/mehuldarak/athena/structure_level_latents/slab_md_unfreeze_li_structure_latents.json"
FPS_SEED_DIR = "/home/mehuldarak/athena/structure_level_latents/fps_seed"
OUTPUT_DIR   = "/home/mehuldarak/athena/structure_level_latents/fps_coverage_trial0"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── PCA config ────────────────────────────────────────────────────────────────
USE_PCA      = True
PCA_VAR      = 0.95   # keep 95 % explained variance
PCA_MIN_DIMS = 32

print("Config ready. Output →", OUTPUT_DIR)
"""))

# ── 2 LABELS ─────────────────────────────────────────────────────────────────
cells.append(md("## 2. Label Definitions\n\nFilenames are matched by **basename** only (path-agnostic)."))
cells.append(code("""\
T_NAMES = {
    "Li_110_slab__LLZO_001_Zr_code93_sto_bestgap_2.50A_r_T550K_3675.cif",
    "Li_100_slab__LLZO_011_La_code71_sto_bestgap_3.00A_r_T550K_25.cif",
    "Li_100_slab__LLZO_110_Li_order17_off_bestgap_2.50A_r_T1100K_75.cif",
    "Li_100_slab__LLZO_010_La_order0_off_bestgap_2.50A_r_T1100K_2200.cif",
    "Li_100_slab__LLZO_010_La_order0_off_bestgap_2.50A_r_T1100K_3800.cif",
    "Li_100_slab__LLZO_001_Zr_code93_sto_bestgap_3.00A_r_T550K_125.cif",
}

V_NAMES = {
    "Li_100_slab__LLZO_001_Zr_code93_sto_bestgap_3.00A_r_T1100K_12100.cif",
    "Li_100_slab__LLZO_010_La_order0_off_bestgap_2.50A_r_T1100K_350.cif",
}

# Te were provided without .cif extension – match both forms
Te_PREFIXES = {
    "Li_110_slab__LLZO_001_Zr_code93_sto_bestgap_2.50A_r_T1100K_200",
    "Li_110_slab__LLZO_010_La_order0_off_bestgap_2.00A_r_T1100K_4775",
}

# All 20 fps_seed stems (reads from the fps_seed directory)
FPS_SEED_NAMES = set(os.listdir(FPS_SEED_DIR))
print(f"fps_seed has {len(FPS_SEED_NAMES)} files")

def get_label(full_key: str) -> str:
    stem = os.path.basename(full_key)
    if stem in T_NAMES:
        return "T"
    if stem in V_NAMES:
        return "V"
    for prefix in Te_PREFIXES:
        if stem == prefix or stem == prefix + ".cif":
            return "Te"
    if stem in FPS_SEED_NAMES:
        return "FPS_other"
    return "pool"
"""))

# ── 3 LOAD ───────────────────────────────────────────────────────────────────
cells.append(md("## 3. Load Latents"))
cells.append(code("""\
print("Loading JSON files…")
with open(JSON_1) as f: data1 = json.load(f)
with open(JSON_2) as f: data2 = json.load(f)

all_data = {**data1, **data2}
print(f"  Total structures: {len(all_data)}  ({len(data1)} maceTrial0 + {len(data2)} slab_md_unfreeze_li)")

filenames  = list(all_data.keys())
labels_arr = np.array([get_label(k) for k in filenames])
embeddings = np.array([all_data[k] for k in filenames], dtype=np.float32)

cnt = pd.Series(labels_arr).value_counts()
print("\\nLabel counts:")
print(cnt.to_string())

# Sanity: all 20 fps_seed names should be labelled
unlabelled_seeds = [n for n in FPS_SEED_NAMES if n not in
                    {os.path.basename(k) for k in all_data}]
if unlabelled_seeds:
    print("\\n⚠ fps_seed files NOT found in latent JSONs:", unlabelled_seeds)
else:
    print("\\n✓ All fps_seed structures found in latent JSONs")
"""))

# ── 4 NORMALISE + PCA ────────────────────────────────────────────────────────
cells.append(md("## 4. L2 Normalise + PCA\n\nPCA is run in the high-dim space for FPS/coverage distances.  \nA separate 2-D PCA is used purely for visualisation."))
cells.append(code("""\
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
emb_norm = (embeddings / norms).astype(np.float32)
print(f"L2-normalised shape: {emb_norm.shape}")

if USE_PCA:
    pca_full = PCA().fit(emb_norm)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = max(PCA_MIN_DIMS, int(np.searchsorted(cumvar, PCA_VAR)) + 1)
    print(f"PCA → {n_comp} components capture {PCA_VAR*100:.0f}% variance")
    pca_hd = PCA(n_components=n_comp)
    emb_hd = pca_hd.fit_transform(emb_norm).astype(np.float32)
else:
    emb_hd = emb_norm
    print("PCA skipped – using raw normalised embeddings")

# 2-D for plotting only
pca_2d = PCA(n_components=2)
emb_2d = pca_2d.fit_transform(emb_norm)
var_2d = pca_2d.explained_variance_ratio_ * 100
print(f"2-D PCA: PC1 = {var_2d[0]:.1f}%,  PC2 = {var_2d[1]:.1f}%")
"""))

# ── 5 INDICES ────────────────────────────────────────────────────────────────
cells.append(md("## 5. Group Indices"))
cells.append(code("""\
idx = {lbl: np.where(labels_arr == lbl)[0]
       for lbl in ("T", "Te", "V", "FPS_other", "pool")}

print("Group sizes:")
for g, arr in idx.items():
    print(f"  {g:<12} {len(arr)}")

emb_T = emb_hd[idx["T"]]   # used as the reference set
"""))

# ── 6 COVERAGE ───────────────────────────────────────────────────────────────
cells.append(md("## 6. Coverage Analysis\n\nFor every structure in V, Te, FPS_other and pool:  \ncompute the **minimum L2 distance to the nearest T point** in the high-dim (PCA) space.  \nExpress as **percentile** of the pool's distance distribution → tells how well T covers it."))
cells.append(code("""\
def nearest_T_dists(group_idx):
    if len(group_idx) == 0 or len(emb_T) == 0:
        return np.array([])
    return cdist(emb_hd[group_idx], emb_T).min(axis=1)

dist = {g: nearest_T_dists(idx[g]) for g in ("V", "Te", "FPS_other", "pool")}

def coverage_percentile(query_dists):
    bg = dist["pool"]
    if len(bg) == 0:
        return np.zeros(len(query_dists))
    return np.array([float(np.mean(bg <= d)) * 100 for d in query_dists])

pct = {g: coverage_percentile(dist[g]) for g in ("V", "Te", "FPS_other")}

print("V   → nearest T:", np.round(dist["V"],  4))
print("Te  → nearest T:", np.round(dist["Te"], 4))
print()
print("V   percentile:", np.round(pct["V"],  1))
print("Te  percentile:", np.round(pct["Te"], 1))
print()
print("FPS_other → nearest T:", np.round(dist["FPS_other"], 4))
print("FPS_other percentile:  ", np.round(pct["FPS_other"], 1))
"""))

# ── 7 COVERAGE TABLE ─────────────────────────────────────────────────────────
cells.append(md("## 7. Coverage Table\n\n> **Low percentile** = well covered by T (gap already in training data)  \n> **High percentile** = large gap → needs more training data in that region"))
cells.append(code("""\
rows = []
for g in ("V", "Te", "FPS_other"):
    for ii, gi in enumerate(idx[g]):
        rows.append({
            "Label":               g,
            "Stem":                os.path.basename(filenames[gi]),
            "Dist_to_nearest_T":   dist[g][ii],
            "Percentile_vs_pool":  pct[g][ii],
        })

df_cov = (pd.DataFrame(rows)
            .sort_values("Percentile_vs_pool", ascending=False)
            .reset_index(drop=True))

# Style the table
df_cov.style.background_gradient(subset=["Percentile_vs_pool"], cmap="RdYlGn_r")
"""))

cells.append(code("""\
# Plain print for terminal / nbconvert compatibility
print(df_cov.to_string(index=False))
"""))

# ── 8 STYLE DICT ─────────────────────────────────────────────────────────────
cells.append(md("## 8. Style Config (shared across plots)"))
cells.append(code("""\
STYLE = {
    "pool":      dict(c="#C0C0C0", marker="o", s=7,   alpha=0.30, zorder=1,
                      label="Pool (~5277)"),
    "FPS_other": dict(c="#9467BD", marker="o", s=75,  alpha=0.85, zorder=3,
                      label="FPS_other — next AL candidates (10)"),
    "T":         dict(c="#1F77B4", marker="*", s=230, alpha=1.00, zorder=5,
                      label="T — finetuning set (6)"),
    "Te":        dict(c="#FF7F0E", marker="^", s=140, alpha=1.00, zorder=5,
                      label="Te — test (2)"),
    "V":         dict(c="#2CA02C", marker="D", s=140, alpha=1.00, zorder=5,
                      label="V — validation (2)"),
}

def annotate(ax, gi, lbl, extra=""):
    stem = os.path.basename(filenames[gi])[-30:]
    txt = ax.annotate(
        f"{lbl}: …{stem}{extra}",
        xy=(emb_2d[gi, 0], emb_2d[gi, 1]),
        xytext=(6, 6), textcoords="offset points",
        fontsize=6.5, color=STYLE[lbl]["c"],
        arrowprops=dict(arrowstyle="->", color=STYLE[lbl]["c"], lw=0.7),
    )
    txt.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])

print("Style config ready.")
"""))

# ── 9 PLOT 1 ─────────────────────────────────────────────────────────────────
cells.append(md("## 9. Plot 1 — Overview: PCA scatter + Distance histogram"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("maceTrial0 Latent Space — Active-Learning Iteration 0",
             fontsize=14, fontweight="bold", y=1.01)

# ── left: PCA scatter ────────────────────────────────────────────────────────
ax = axes[0]
ax.set_title("PCA 2-D — Pool, FPS seeds, T / V / Te", fontsize=12)

ax.scatter(emb_2d[idx["pool"], 0],      emb_2d[idx["pool"], 1],      **STYLE["pool"])
ax.scatter(emb_2d[idx["FPS_other"], 0], emb_2d[idx["FPS_other"], 1], **STYLE["FPS_other"])
for lbl in ("T", "Te", "V"):
    if len(idx[lbl]):
        ax.scatter(emb_2d[idx[lbl], 0], emb_2d[idx[lbl], 1], **STYLE[lbl])
        for gi in idx[lbl]:
            annotate(ax, gi, lbl)

ax.set_xlabel(f"PC1 ({var_2d[0]:.1f}%)", fontsize=10)
ax.set_ylabel(f"PC2 ({var_2d[1]:.1f}%)", fontsize=10)
ax.legend(fontsize=8, loc="best")
ax.grid(True, alpha=0.25)

# ── right: distance histogram ────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_title("Distance to Nearest T — Coverage in Finetuned Latent Space", fontsize=12)
if len(dist["pool"]):
    ax2.hist(dist["pool"], bins=70, color="#C0C0C0", alpha=0.55,
             density=True, label="Pool → nearest T")

colours = {"V": STYLE["V"]["c"], "Te": STYLE["Te"]["c"],
           "FPS_other": STYLE["FPS_other"]["c"]}
for g in ("V", "Te", "FPS_other"):
    for d in dist[g]:
        ax2.axvline(d, color=colours[g], lw=1.8, linestyle="--",
                    label=f"{g} d={d:.3f}")

ax2.set_xlabel("Min distance to nearest T point", fontsize=10)
ax2.set_ylabel("Density", fontsize=10)
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fps_coverage_overview.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("Plot 1 saved.")
"""))

# ── 10 PLOT 2 ────────────────────────────────────────────────────────────────
cells.append(md("## 10. Plot 2 — T Coverage Circles\n\nEach T point gets a shaded circle whose radius = half the nearest-T gap  \n(i.e. approximate coverage radius in the 2-D projection).  \nV / Te labels carry their percentile."))
cells.append(code("""\
fig2, ax3 = plt.subplots(figsize=(11, 8))
ax3.set_title("maceTrial0 Latent Space — T Coverage Circles\\n"
              "(shaded radius = half nearest-T gap in PCA-reduced space)",
              fontsize=12, fontweight="bold")

ax3.scatter(emb_2d[idx["pool"], 0],      emb_2d[idx["pool"], 1],      **STYLE["pool"])
ax3.scatter(emb_2d[idx["FPS_other"], 0], emb_2d[idx["FPS_other"], 1], **STYLE["FPS_other"])

# Coverage radius per T point
if len(idx["T"]) >= 2:
    D_TT = cdist(emb_hd[idx["T"]], emb_hd[idx["T"]])
    np.fill_diagonal(D_TT, np.inf)
    r_T = D_TT.min(axis=1) / 2.0
else:
    r_T = np.full(len(idx["T"]), 0.05)

for ii, gi in enumerate(idx["T"]):
    ax3.add_patch(plt.Circle(
        (emb_2d[gi, 0], emb_2d[gi, 1]), radius=r_T[ii],
        color=STYLE["T"]["c"], alpha=0.13, zorder=2
    ))

for lbl in ("T", "V", "Te"):
    if len(idx[lbl]):
        ax3.scatter(emb_2d[idx[lbl], 0], emb_2d[idx[lbl], 1], **STYLE[lbl])
        for k, gi in enumerate(idx[lbl]):
            extra = ""
            if lbl in ("V", "Te"):
                row = df_cov[df_cov["Stem"] == os.path.basename(filenames[gi])]
                if len(row):
                    extra = f"  [{row.iloc[0]['Percentile_vs_pool']:.0f}th pct]"
            annotate(ax3, gi, lbl, extra)

ax3.set_xlabel(f"PC1 ({var_2d[0]:.1f}%)", fontsize=10)
ax3.set_ylabel(f"PC2 ({var_2d[1]:.1f}%)", fontsize=10)
ax3.legend(fontsize=9, loc="best")
ax3.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fps_coverage_T_radii.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("Plot 2 saved.")
"""))

# ── 11 PLOT 3 ────────────────────────────────────────────────────────────────
cells.append(md("## 11. Plot 3 — CDF Percentile Plot\n\nThe grey curve is the pool's empirical CDF of distance to nearest T.  \nVertical dashed lines mark where V / Te / FPS_other sit on that distribution."))
cells.append(code("""\
if len(dist["pool"]):
    sorted_pool = np.sort(dist["pool"])
    cdf_y = np.linspace(0, 100, len(sorted_pool))

    fig3, ax4 = plt.subplots(figsize=(10, 5))
    ax4.set_title("CDF: Pool → Nearest T  |  V / Te / FPS_other Markers",
                  fontsize=12, fontweight="bold")

    ax4.plot(sorted_pool, cdf_y, color="#A0A0A0", lw=2, label="Pool CDF")
    ax4.fill_betweenx(cdf_y, sorted_pool, alpha=0.08, color="#A0A0A0")

    for g in ("V", "Te", "FPS_other"):
        c = STYLE[g]["c"]
        mk = STYLE[g]["marker"]
        for d, p in zip(dist[g], pct[g]):
            ax4.axvline(d, color=c, lw=1.6, linestyle="--",
                        label=f"{g}  d={d:.3f}  → {p:.0f}th pct")
            ax4.axhline(p, color=c, lw=0.8, linestyle=":")
            ax4.scatter([d], [p], color=c, marker=mk, s=80, zorder=6)

    ax4.set_xlabel("Min distance to nearest T (maceTrial0 latent space)", fontsize=10)
    ax4.set_ylabel("% of pool points closer to T", fontsize=10)
    ax4.legend(fontsize=7, ncol=2, loc="upper left")
    ax4.grid(True, alpha=0.25)
    ax4.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fps_coverage_percentile_cdf.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot 3 saved.")
else:
    print("Skipped — no pool points.")
"""))

# ── 12 SUMMARY ───────────────────────────────────────────────────────────────
cells.append(md("## 12. Summary Table & CSV Export"))
cells.append(code("""\
print("══ COVERAGE SUMMARY (maceTrial0) ══")
print(f"{'Group':<12} {'N':>4}  {'Mean dist → T':>14}  {'Mean pct':>10}")
print("─" * 46)
for g in ("V", "Te", "FPS_other"):
    d_g, p_g = dist[g], pct[g]
    if len(d_g):
        print(f"{g:<12} {len(d_g):>4}  {d_g.mean():>14.4f}  {p_g.mean():>9.1f}%")
print("═" * 46)
print()
print("Per-structure detail:")
print(df_cov.to_string(index=False))

# Save CSVs
df_cov.to_csv(os.path.join(OUTPUT_DIR, "coverage_summary.csv"), index=False)
print("\\nSaved: coverage_summary.csv")

# All 20 fps_seed stems with labels
fps_rows = []
for g in ("T", "V", "Te", "FPS_other"):
    for k, gi in enumerate(idx[g]):
        r = {"Label": g, "Stem": os.path.basename(filenames[gi]),
             "Filename": filenames[gi]}
        if g != "T":
            r["Dist_to_nearest_T"] = dist[g][k]
            r["Percentile_vs_pool"] = pct[g][k]
        fps_rows.append(r)
df_fps_seed = pd.DataFrame(fps_rows)
df_fps_seed.to_csv(os.path.join(OUTPUT_DIR, "fps_seeds_coverage.csv"), index=False)
print("Saved: fps_seeds_coverage.csv")
df_fps_seed
"""))

# ── BUILD NOTEBOOK ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "mace",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook written to: {NB_PATH}")
print(f"Cells: {len(cells)}")
