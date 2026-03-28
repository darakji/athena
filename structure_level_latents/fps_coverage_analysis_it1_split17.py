"""
fps_coverage_analysis_it1_split17.py
-------------------------------------
Iteration 1 coverage analysis — mace_fps_split17 model.

Dataset roles
─────────────
  T   (seed / reference) : 16 training structures used to train mace_fps_split17
                           → it0_TrainingEmbs_split17_structure_latents.json
  pool                   : ~1200 structures produced by MD with the split-17 model
                           → it0_split17_structure_latents.json

Analysis mirrors fps_coverage_analysis_trial0.ipynb:
  • L2-normalise → PCA (95% var, min 32 dims) for distance metrics
  • PCA-2D for visualisation
  • Per-group nearest-T distances & percentiles vs pool
  • 3 plots + CSV export
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

# ── Paths ─────────────────────────────────────────────────────────────────────
POOL_JSON = (
    "/home/mehuldarak/athena/structure_level_latents/"
    "iteration0_mace_latents/it0_split17_structure_latents.json"
)
SEED_JSON = (
    "/home/mehuldarak/athena/structure_level_latents/"
    "iteration0_mace_latents/it0_TrainingEmbs_split17_structure_latents.json"
)
OUTPUT_DIR = (
    "/home/mehuldarak/athena/structure_level_latents/"
    "fps_coverage_it1_split17"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── PCA config ────────────────────────────────────────────────────────────────
USE_PCA   = True
PCA_VAR   = 0.95
PCA_MIN_DIMS = 32

print("Config ready.  Output →", OUTPUT_DIR)

# =============================================================================
# 1. Load latents
# =============================================================================
print("\nLoading JSON files…")
with open(POOL_JSON) as f:
    pool_data = json.load(f)
with open(SEED_JSON) as f:
    seed_data = json.load(f)

print(f"  Pool  : {len(pool_data)} structures  (MD pool from mace_fps_split17)")
print(f"  Seed  : {len(seed_data)} structures  (training set of split-17 → role = T)")

# =============================================================================
# 2. Build unified arrays with labels
#    T   = 16 training structures (seed)
#    pool = the MD-generated pool (minus any overlap with T)
# =============================================================================
seed_basenames = {os.path.basename(k) for k in seed_data}

filenames  = []
labels_raw = []
emb_list   = []

# Add seed first (T)
for k, v in seed_data.items():
    filenames.append(k)
    labels_raw.append("T")
    emb_list.append(v)

# Add pool — mark as pool (skip if same basename as seed to avoid duplicates)
for k, v in pool_data.items():
    bn = os.path.basename(k)
    if bn in seed_basenames:
        continue          # skip exact basename overlap
    filenames.append(k)
    labels_raw.append("pool")
    emb_list.append(v)

labels_arr = np.array(labels_raw)
embeddings  = np.array(emb_list, dtype=np.float32)

cnt = pd.Series(labels_arr).value_counts()
print("\nLabel counts after merge:")
print(cnt.to_string())

# =============================================================================
# 3. L2-normalise + PCA
# =============================================================================
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
emb_norm = (embeddings / norms).astype(np.float32)
print(f"\nL2-normalised shape: {emb_norm.shape}")

if USE_PCA:
    pca_full = PCA().fit(emb_norm)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp   = max(PCA_MIN_DIMS, int(np.searchsorted(cumvar, PCA_VAR)) + 1)
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

# =============================================================================
# 4. Group indices
# =============================================================================
idx = {lbl: np.where(labels_arr == lbl)[0]
       for lbl in ("T", "pool")}

print("\nGroup sizes:")
for g, arr in idx.items():
    print(f"  {g:<12} {len(arr)}")

emb_T = emb_hd[idx["T"]]   # reference set

# =============================================================================
# 5. Distance computations  (pool → nearest T)
# =============================================================================
print("\nComputing distances to nearest T…")

def nearest_T_dist(group_embs):
    """Min cosine distance from each structure to its nearest T."""
    if len(group_embs) == 0:
        return np.array([])
    D = cdist(group_embs, emb_T, metric="euclidean")
    return D.min(axis=1)

dist_pool = nearest_T_dist(emb_hd[idx["pool"]])

# Percentile of each pool structure relative to the whole pool distribution
def pool_percentile(d_vals):
    if len(d_vals) == 0:
        return np.array([])
    return np.array([
        100.0 * (dist_pool < d).sum() / len(dist_pool)
        for d in d_vals
    ])

pct_pool = pool_percentile(dist_pool)   # for the pool itself (sanity: uniform)

print(f"  pool  : mean dist = {dist_pool.mean():.4f},  "
      f"median = {np.median(dist_pool):.4f}")

# =============================================================================
# 6. Coverage summary dataframe (pool structures only)
# =============================================================================
pool_stems = [os.path.basename(filenames[i]) for i in idx["pool"]]
df_cov = pd.DataFrame({
    "Stem":              pool_stems,
    "Dist_to_nearest_T": dist_pool,
    "Percentile_vs_pool": pct_pool,
}).sort_values("Dist_to_nearest_T", ascending=False).reset_index(drop=True)

print("\nTop-20 pool structures farthest from T (coverage gaps):")
print(df_cov.head(20).to_string(index=False))

# =============================================================================
# 7. Style config
# =============================================================================
STYLE = {
    "pool": dict(c="#C0C0C0", marker="o", s=7,   alpha=0.30, zorder=1,
                 label=f"Pool ({len(idx['pool'])})"),
    "T":    dict(c="#1F77B4", marker="*", s=230, alpha=1.00, zorder=5,
                 label=f"T — training seed ({len(idx['T'])})"),
}

def annotate(ax, gi, lbl, extra=""):
    stem = os.path.basename(filenames[gi])[-35:]
    txt = ax.annotate(
        f"{lbl}: …{stem}{extra}",
        xy=(emb_2d[gi, 0], emb_2d[gi, 1]),
        xytext=(6, 6), textcoords="offset points",
        fontsize=5.5, color=STYLE[lbl]["c"],
        arrowprops=dict(arrowstyle="->", color=STYLE[lbl]["c"], lw=0.7),
    )
    txt.set_path_effects([
        pe.Stroke(linewidth=2, foreground="white"), pe.Normal()
    ])

print("\nStyle config ready.")

# =============================================================================
# 8. Plot 1 — Overview: PCA scatter + Distance histogram
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "mace_fps_split17 — Iteration 1 Latent Space Coverage Analysis",
    fontsize=13, fontweight="bold"
)

# ── Left: PCA scatter ────────────────────────────────────────────────────────
ax1.set_title("PCA 2D — pool + T seed", fontsize=11)
for lbl in ("pool", "T"):
    if len(idx[lbl]):
        ax1.scatter(emb_2d[idx[lbl], 0], emb_2d[idx[lbl], 1], **STYLE[lbl])

ax1.set_xlabel(f"PC1 ({var_2d[0]:.1f}%)", fontsize=10)
ax1.set_ylabel(f"PC2 ({var_2d[1]:.1f}%)", fontsize=10)
ax1.legend(fontsize=8, loc="best")
ax1.grid(True, alpha=0.25)

# ── Right: histogram of pool distances ───────────────────────────────────────
ax2.set_title("Pool distance to nearest T distribution", fontsize=11)
ax2.hist(dist_pool, bins=60, color="#C0C0C0", alpha=0.5, label="pool")
ax2.set_xlabel("Min distance to nearest T (PCA space)", fontsize=10)
ax2.set_ylabel("Count", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)

plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "it1_split17_coverage_overview.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot 1 saved → {out1}")

# =============================================================================
# 9. Plot 2 — T Coverage Circles
# =============================================================================
fig2, ax3 = plt.subplots(figsize=(12, 9))
ax3.set_title(
    "mace_fps_split17 — T Coverage Circles\n"
    "(shaded radius = half nearest-T gap, PCA-reduced space)",
    fontsize=12, fontweight="bold"
)

ax3.scatter(emb_2d[idx["pool"], 0], emb_2d[idx["pool"], 1], **STYLE["pool"])

# Coverage radii from inter-T distances
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

ax3.scatter(emb_2d[idx["T"], 0], emb_2d[idx["T"], 1], **STYLE["T"])
for gi in idx["T"]:
    annotate(ax3, gi, "T")

ax3.set_xlabel(f"PC1 ({var_2d[0]:.1f}%)", fontsize=10)
ax3.set_ylabel(f"PC2 ({var_2d[1]:.1f}%)", fontsize=10)
ax3.legend(fontsize=9, loc="best")
ax3.grid(True, alpha=0.25)

plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "it1_split17_coverage_T_radii.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot 2 saved → {out2}")

# =============================================================================
# 10. Plot 3 — CDF of pool distances to T
# =============================================================================
if len(dist_pool):
    sorted_pool = np.sort(dist_pool)
    cdf_y = np.linspace(0, 100, len(sorted_pool))

    fig3, ax4 = plt.subplots(figsize=(10, 5))
    ax4.set_title(
        "CDF: Pool → Nearest T  (mace_fps_split17 latent space, Iteration 1)",
        fontsize=12, fontweight="bold"
    )
    ax4.plot(sorted_pool, cdf_y, color="#A0A0A0", lw=2, label="Pool CDF")
    ax4.fill_betweenx(cdf_y, sorted_pool, alpha=0.08, color="#A0A0A0")

    # Mark the 50th, 75th, 90th, 95th percentiles
    for pct_mark, ls in [(50, "--"), (75, "-."), (90, ":"), (95, ":")]:
        d_at_pct = np.percentile(dist_pool, pct_mark)
        ax4.axvline(d_at_pct, color="steelblue", lw=1.2, linestyle=ls,
                    label=f"{pct_mark}th pct  d={d_at_pct:.4f}")

    ax4.set_xlabel(
        "Min distance to nearest T (mace_fps_split17 latent space)", fontsize=10
    )
    ax4.set_ylabel("% of pool points closer to T", fontsize=10)
    ax4.legend(fontsize=8, loc="upper left")
    ax4.grid(True, alpha=0.25)
    ax4.set_xlim(left=0)

    plt.tight_layout()
    out3 = os.path.join(OUTPUT_DIR, "it1_split17_coverage_percentile_cdf.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot 3 saved → {out3}")
else:
    print("Skipped CDF plot — no pool distances.")

# =============================================================================
# 11. Summary + CSV export
# =============================================================================
print("\n══ COVERAGE SUMMARY (iteration 1 / split-17) ══")
print(f"  Pool structures  : {len(dist_pool)}")
print(f"  T seed structures: {len(idx['T'])}")
print(f"  Mean dist → T    : {dist_pool.mean():.4f}")
print(f"  Median dist → T  : {np.median(dist_pool):.4f}")
print(f"  90th pct dist    : {np.percentile(dist_pool, 90):.4f}")
print(f"  95th pct dist    : {np.percentile(dist_pool, 95):.4f}")
print("═" * 50)

csv_path = os.path.join(OUTPUT_DIR, "it1_split17_coverage_summary.csv")
df_cov.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# Also save T info
t_rows = [{"Label": "T", "Stem": os.path.basename(filenames[gi]),
           "Filename": filenames[gi]} for gi in idx["T"]]
df_T = pd.DataFrame(t_rows)
t_csv = os.path.join(OUTPUT_DIR, "it1_split17_T_seed.csv")
df_T.to_csv(t_csv, index=False)
print(f"Saved: {t_csv}")

print("\nDone.")
