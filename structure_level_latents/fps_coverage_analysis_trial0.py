"""
fps_coverage_analysis_trial0.py
=================================
Active-Learning Iteration Analysis — Trial 0

Context
-------
• A universal MACE model was used to run FPS over the full pool → 20 "fps_seed" structures.
• 6 of those 20 were used as T (training) to finetune MACE → maceTrial0.
• 2 of those 20 are V (validation), 2 are Te (test).
• The remaining 10 FPS seeds are unlabelled candidates for the next AL iteration.

This script operates on maceTrial0's latent space and asks:
  "In the finetuned model's representation, how well does T cover V, Te,
   and the rest of the pool?"

Labels
------
  T        (6)  : training structures used for finetuning
  V        (2)  : validation structures
  Te       (2)  : test structures
  FPS_other(10) : other fps_seed candidates (not used in this iteration)
  pool        : everything else (~5277)

Outputs (in OUTPUT_DIR)
-----------------------
  fps_coverage_overview.png       — PCA scatter + distance histogram
  fps_coverage_T_radii.png        — PCA with T coverage circles
  fps_coverage_percentile_cdf.png — CDF with V/Te/FPS_other markers
  fps_selection.csv               — all 20 FPS seeds with labels & coverage
  coverage_summary.csv            — V/Te/FPS_other coverage vs T
"""

import json
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
JSON_1 = "/home/mehuldarak/athena/structure_level_latents/maceTrial0_structure_latents.json"
JSON_2 = "/home/mehuldarak/athena/structure_level_latents/slab_md_unfreeze_li_structure_latents.json"
FPS_SEED_DIR = "/home/mehuldarak/athena/structure_level_latents/fps_seed"

OUTPUT_DIR = "/home/mehuldarak/athena/structure_level_latents/fps_coverage_trial0"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_PCA = True
PCA_VAR = 0.95
PCA_MIN_DIMS = 32

# ══════════════════════════════════════════════════════════════════════════════
# LABEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

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
# Te stems were given without .cif → match by prefix
Te_PREFIXES = {
    "Li_110_slab__LLZO_001_Zr_code93_sto_bestgap_2.50A_r_T1100K_200",
    "Li_110_slab__LLZO_010_La_order0_off_bestgap_2.00A_r_T1100K_4775",
}

# All 20 fps_seed file stems (from ls fps_seed/)
FPS_SEED_NAMES = {os.path.basename(f) for f in os.listdir(FPS_SEED_DIR)}


def get_label(full_key: str) -> str:
    stem = os.path.basename(full_key)
    if stem in T_NAMES:
        return "T"
    if stem in V_NAMES:
        return "V"
    # Te: prefix match (given without .cif)
    for prefix in Te_PREFIXES:
        if stem == prefix or stem == prefix + ".cif":
            return "Te"
    if stem in FPS_SEED_NAMES:
        return "FPS_other"
    return "pool"


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading latents…")
with open(JSON_1) as f:
    data1 = json.load(f)
with open(JSON_2) as f:
    data2 = json.load(f)

all_data = {**data1, **data2}
print(f"  Total: {len(all_data)}  ({len(data1)} maceTrial0 + {len(data2)} slab_md)")

filenames  = list(all_data.keys())
labels_arr = np.array([get_label(k) for k in filenames])
embeddings = np.array([all_data[k] for k in filenames], dtype=np.float32)

print(pd.Series(labels_arr).value_counts().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# L2 NORMALISE + PCA
# ══════════════════════════════════════════════════════════════════════════════
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
emb_norm = (embeddings / norms).astype(np.float32)

if USE_PCA:
    pca_full = PCA()
    pca_full.fit(emb_norm)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = max(PCA_MIN_DIMS, int(np.searchsorted(cumvar, PCA_VAR)) + 1)
    print(f"PCA: {n_comp} components for {PCA_VAR*100:.0f}% variance")
    pca = PCA(n_components=n_comp)
    emb_fps = pca.fit_transform(emb_norm).astype(np.float32)
else:
    emb_fps = emb_norm

# 2-D for visualisation
pca_2d = PCA(n_components=2)
emb_2d = pca_2d.fit_transform(emb_norm)

# ══════════════════════════════════════════════════════════════════════════════
# INDEX GROUPS
# ══════════════════════════════════════════════════════════════════════════════
idx = {lbl: np.where(labels_arr == lbl)[0]
       for lbl in ("T", "Te", "V", "FPS_other", "pool")}

emb_T = emb_fps[idx["T"]]
print(f"\nGroup sizes: T={len(idx['T'])}  V={len(idx['V'])}  "
      f"Te={len(idx['Te'])}  FPS_other={len(idx['FPS_other'])}  "
      f"pool={len(idx['pool'])}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: dist to nearest T
# ══════════════════════════════════════════════════════════════════════════════
def nearest_T_dists(group_idx):
    if len(group_idx) == 0 or len(emb_T) == 0:
        return np.array([])
    return cdist(emb_fps[group_idx], emb_T).min(axis=1)


dist = {g: nearest_T_dists(idx[g])
        for g in ("V", "Te", "FPS_other", "pool")}


def coverage_percentile(query_dists):
    bg = dist["pool"]
    if len(bg) == 0:
        return np.zeros(len(query_dists))
    return np.array([float(np.mean(bg <= d)) * 100 for d in query_dists])


pct = {g: coverage_percentile(dist[g]) for g in ("V", "Te", "FPS_other")}

# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════
rows_cov = []
for g in ("V", "Te", "FPS_other"):
    for ii, global_i in enumerate(idx[g]):
        rows_cov.append({
            "Label": g,
            "Stem": os.path.basename(filenames[global_i]),
            "Dist_to_nearest_T": dist[g][ii],
            "Percentile_vs_pool": pct[g][ii],
        })
df_cov = pd.DataFrame(rows_cov).sort_values("Percentile_vs_pool", ascending=False)
print("\n── Coverage of V / Te / FPS_other by T (maceTrial0 latent space) ──")
print(df_cov.to_string(index=False))
print("\nNote: percentile = X  →  X% of pool points are closer to T than this structure.")
print("      Low  = well covered by T  |  High = gap in training data")

# Also rows for T itself (for FPS seed table)
rows_fps = []
for g in ("T", "V", "Te", "FPS_other"):
    for global_i in idx[g]:
        row = {
            "Label": g,
            "Stem": os.path.basename(filenames[global_i]),
            "Filename": filenames[global_i],
        }
        if g != "T":
            ii = list(idx[g]).index(global_i)
            row["Dist_to_nearest_T"] = dist[g][ii]
            row["Percentile_vs_pool"] = pct[g][ii]
        rows_fps.append(row)
df_fps = pd.DataFrame(rows_fps)

# ══════════════════════════════════════════════════════════════════════════════
# VISUALS
# ══════════════════════════════════════════════════════════════════════════════
STYLE = {
    "pool":      dict(c="#C0C0C0", marker="o", s=7,   alpha=0.30, zorder=1, label="Pool"),
    "FPS_other": dict(c="#9467BD", marker="o", s=70,  alpha=0.80, zorder=3, label="FPS_other (next AL candidates)"),
    "T":         dict(c="#1F77B4", marker="*", s=220, alpha=1.00, zorder=5, label="T — finetuning set (6)"),
    "Te":        dict(c="#FF7F0E", marker="^", s=130, alpha=1.00, zorder=5, label="Te — test (2)"),
    "V":         dict(c="#2CA02C", marker="D", s=130, alpha=1.00, zorder=5, label="V — validation (2)"),
}

def add_labels(ax, group_indices, lbl, extra_str=""):
    for global_i in group_indices:
        stem = os.path.basename(filenames[global_i])[-32:]
        txt = ax.annotate(
            f"{lbl}: …{stem}{extra_str}",
            xy=(emb_2d[global_i, 0], emb_2d[global_i, 1]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=6.5, color=STYLE[lbl]["c"],
            arrowprops=dict(arrowstyle="->", color=STYLE[lbl]["c"], lw=0.7),
        )
        txt.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])


# ── Plot 1: overview ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("maceTrial0 Latent Space — Active Learning Iteration 0",
             fontsize=14, fontweight="bold", y=1.01)

ax = axes[0]
ax.set_title("PCA 2-D — Pool, FPS seeds, T / V / Te", fontsize=12)
ax.scatter(emb_2d[idx["pool"], 0], emb_2d[idx["pool"], 1], **STYLE["pool"])
ax.scatter(emb_2d[idx["FPS_other"], 0], emb_2d[idx["FPS_other"], 1], **STYLE["FPS_other"])
for lbl in ("T", "Te", "V"):
    if len(idx[lbl]):
        ax.scatter(emb_2d[idx[lbl], 0], emb_2d[idx[lbl], 1], **STYLE[lbl])
# Annotate non-pool
add_labels(ax, idx["T"], "T")
add_labels(ax, idx["V"], "V")
add_labels(ax, idx["Te"], "Te")

ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
ax.legend(fontsize=8, loc="best")
ax.grid(True, alpha=0.25)

ax2 = axes[1]
ax2.set_title("Distance to Nearest T — Coverage in Finetuned Latent Space", fontsize=12)
if len(dist["pool"]):
    ax2.hist(dist["pool"], bins=70, color="#C0C0C0", alpha=0.6,
             density=True, label="Pool → nearest T")
for g, col in [("V", STYLE["V"]["c"]), ("Te", STYLE["Te"]["c"]),
               ("FPS_other", STYLE["FPS_other"]["c"])]:
    for d in dist[g]:
        ax2.axvline(d, color=col, lw=1.8, linestyle="--",
                    label=f"{g} (d={d:.3f})")
ax2.set_xlabel("Min distance to nearest T point", fontsize=10)
ax2.set_ylabel("Density", fontsize=10)
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True, alpha=0.25)

plt.tight_layout()
p1 = os.path.join(OUTPUT_DIR, "fps_coverage_overview.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {p1}")

# ── Plot 2: T coverage circles + FPS_other ────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(11, 8))
ax3.set_title("maceTrial0 Latent Space — T Coverage Circles\n"
              "Shaded region = estimated T coverage (half nearest-T gap)",
              fontsize=12, fontweight="bold")

ax3.scatter(emb_2d[idx["pool"], 0], emb_2d[idx["pool"], 1], **STYLE["pool"])
ax3.scatter(emb_2d[idx["FPS_other"], 0], emb_2d[idx["FPS_other"], 1], **STYLE["FPS_other"])

# Coverage radius: half the distance to the nearest other T point (in reduced space)
if len(idx["T"]) >= 2:
    D_T_T = cdist(emb_fps[idx["T"]], emb_fps[idx["T"]])
    np.fill_diagonal(D_T_T, np.inf)
    r_T = D_T_T.min(axis=1) / 2.0
else:
    r_T = np.full(len(idx["T"]), 0.05)

for ii, gi in enumerate(idx["T"]):
    ax3.add_patch(plt.Circle(
        (emb_2d[gi, 0], emb_2d[gi, 1]), radius=r_T[ii],
        color=STYLE["T"]["c"], alpha=0.12, zorder=2
    ))

for lbl in ("T", "V", "Te"):
    if len(idx[lbl]):
        ax3.scatter(emb_2d[idx[lbl], 0], emb_2d[idx[lbl], 1], **STYLE[lbl])

# Annotate V and Te with percentile
for g in ("T", "V", "Te"):
    for k, gi in enumerate(idx[g]):
        stem = os.path.basename(filenames[gi])[-30:]
        extra = ""
        if g in ("V", "Te"):
            row = df_cov[df_cov["Stem"] == os.path.basename(filenames[gi])]
            if len(row):
                extra = f"  [{row.iloc[0]['Percentile_vs_pool']:.0f}th pct]"
        txt = ax3.annotate(
            f"{g}: …{stem}{extra}",
            xy=(emb_2d[gi, 0], emb_2d[gi, 1]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=7, color=STYLE[g]["c"],
            arrowprops=dict(arrowstyle="->", color=STYLE[g]["c"], lw=0.8)
        )
        txt.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])

ax3.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
ax3.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
ax3.legend(fontsize=9, loc="best")
ax3.grid(True, alpha=0.25)

p2 = os.path.join(OUTPUT_DIR, "fps_coverage_T_radii.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p2}")

# ── Plot 3: CDF percentile ────────────────────────────────────────────────────
if len(dist["pool"]):
    sorted_pool = np.sort(dist["pool"])
    pct_axis = np.linspace(0, 100, len(sorted_pool))

    fig3, ax4 = plt.subplots(figsize=(10, 5))
    ax4.set_title("CDF: Pool → Nearest T  |  V / Te / FPS_other markers",
                  fontsize=12, fontweight="bold")
    ax4.plot(sorted_pool, pct_axis, color="#A0A0A0", lw=2, label="Pool CDF")
    ax4.fill_betweenx(pct_axis, sorted_pool, alpha=0.08, color="#A0A0A0")

    for g, style_g in [("V", STYLE["V"]), ("Te", STYLE["Te"]),
                        ("FPS_other", STYLE["FPS_other"])]:
        for d, p in zip(dist[g], pct[g]):
            ax4.axvline(d, color=style_g["c"], lw=1.6, linestyle="--",
                        label=f"{g} d={d:.3f}  →  {p:.0f}th pct")
            ax4.axhline(p, color=style_g["c"], lw=0.8, linestyle=":")
            ax4.scatter([d], [p], color=style_g["c"],
                        marker=style_g["marker"], s=80, zorder=6)

    ax4.set_xlabel("Min distance to nearest T point (maceTrial0 latent space)", fontsize=10)
    ax4.set_ylabel("% of pool points closer to T", fontsize=10)
    ax4.legend(fontsize=7, ncol=2, loc="upper left")
    ax4.grid(True, alpha=0.25)
    ax4.set_xlim(left=0)

    p3 = os.path.join(OUTPUT_DIR, "fps_coverage_percentile_cdf.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p3}")

# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORTS
# ══════════════════════════════════════════════════════════════════════════════
cov_csv = os.path.join(OUTPUT_DIR, "coverage_summary.csv")
df_cov.to_csv(cov_csv, index=False)
print(f"Saved: {cov_csv}")

fps_csv = os.path.join(OUTPUT_DIR, "fps_seeds_coverage.csv")
df_fps.to_csv(fps_csv, index=False)
print(f"Saved: {fps_csv}")

print("\n═══ SUMMARY ════════════════════════════════════════════════════════════")
print(f"{'Group':<12} {'N':>4}  {'Mean dist to T':>14}  {'Mean percentile':>16}")
print("─" * 52)
for g in ("V", "Te", "FPS_other"):
    if len(dist[g]):
        print(f"{g:<12} {len(dist[g]):>4}  {dist[g].mean():>14.4f}  {pct[g].mean():>15.1f}%")
print("═" * 52)
print("done.")
