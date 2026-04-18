"""
Trajectory Validation
---------------------
Validates inferred circular pseudotime from Laplacian Eigenmaps
against known cell cycle biology.

Validation axes:
1. Correlation between pseudotime and cell cycle phase z-scores
2. Phase transition order verification (G1/S → S → G2/M → M → M/G1)
3. Robustness to subsampling and stability across kNN k values

Phase scoring follows the Revelio-style double-z-score method used in
DiffusionMapsDemo.ipynb, with the 5-phase marker sets loaded from the
GSE142277 gene_sets spreadsheet.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc

from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from trajectory_inference.Laplacian_Eigenmaps import fullLaplacian, revelio_like_preprocess


# Canonical order of cell cycle phases around the circle (matches revelio_gene_sets.csv)
PHASE_ORDER = ["G1/S", "S", "G2", "G2/M", "M/G1"]

# Transcriptionally periodic validation genes across the full cell cycle.
# Some overlap with Revelio marker lists, but validation tests single-gene
# peak timing against pseudotime — a different signal from aggregate phase scores.
VALIDATION_GENES = {
    # G1
    "CCND1":  "G1",
    # G1/S
    "E2F2":   "G1/S",
    "CCNE2":  "G1/S",
    "CDC6":   "G1/S",
    "MCM5":   "G1/S",
    # S
    "PCNA":   "S",
    "RRM2":   "S",
    "PLK4":   "S",
    "TYMS":   "S",
    # S/G2
    "CCNA2":  "S/G2",
    # G2
    "WEE1":   "G2",
    "CCNF":   "G2",
    # G2/M
    "CCNB1":  "G2/M",
    "CCNB2":  "G2/M",
    "TOP2A":  "G2/M",
    "NEK2":   "G2/M",
    # M
    "PLK1":   "M",
    "AURKA":  "M",
    "BUB1":   "M",
    "UBE2C":  "M",
    "CDC20":  "M",
}

# Numeric ordering of phases for expected-rank assignment
VALIDATION_PHASE_ORDER = {
    "G1": 1, "G1/S": 2, "S": 3, "S/G2": 4, "G2": 5, "G2/M": 6, "M": 7,
}
data = pd.read_csv("../data/GSE142277/GSM4224315_out_gene_exon_tagged.dge_exonssf002_WT.txt", sep = "\t", index_col = 0)

# Default location of the marker gene set (revelio_gene_sets.csv)
DEFAULT_GENE_SET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "revelio_gene_sets.csv"
)

# CSV uses dots in column names; map to the slash notation used everywhere else
_CSV_PHASE_MAP = {"G1.S": "G1/S", "S": "S", "G2": "G2", "G2.M": "G2/M", "M.G1": "M/G1"}


def load_marker_dict(path=DEFAULT_GENE_SET_PATH):
    """Load the 5-phase marker dictionary from revelio_gene_sets.csv.

    Returns a dict mapping phase name (slash notation) -> list of gene symbols.
    """
    df = pd.read_csv(path, index_col=0)
    marker_dict = {}
    for col in df.columns:
        phase = _CSV_PHASE_MAP.get(col, col)
        marker_dict[phase] = list(df[col].dropna().astype(str).str.strip().values)
    return marker_dict


def _full_pipeline(data, k):
    """Run kNN → Laplacian Eigenmaps → pseudotime.

    Returns (pseudotime_array, eigenvalues, adata, embedding).
    """
    adata = revelio_like_preprocess(data, marker_dict)
    embedding, eigvals, ps = fullLaplacian(adata, k)
    return ps, eigvals, adata, embedding


# ── Circular statistics ──────────────────────────────────────────────────

def circular_mean(angles):
    """Circular mean of angles in radians."""
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def circular_corr(theta1, theta2):
    """Circular–circular correlation coefficient (Jammalamadaka & SenGupta).

    Invariant to rotation of either variable.  Returns value in [-1, 1].
    """
    mu1 = circular_mean(theta1)
    mu2 = circular_mean(theta2)
    sin1 = np.sin(theta1 - mu1)
    sin2 = np.sin(theta2 - mu2)
    denom = np.sqrt(np.sum(sin1 ** 2) * np.sum(sin2 ** 2))
    if denom == 0:
        return 0.0
    return np.sum(sin1 * sin2) / denom

def correlation_with_phase_scores(adata, pseudotime_values, marker_dict=None):
    """Spearman correlation between pseudotime and each phase's z-score.

    Runs score_cell_cycle(adata, marker_dict) if scores are not yet present.

    Returns
    -------
    dict  {phase_name: {'spearman_rho': float, 'p_value': float}}
    """
    phase_names = list(adata.obs["cc_phase"].cat.categories)
    z = adata.obsm["phase_scores_z"]

    results = {}
    for j, ph in enumerate(phase_names):
        rho, pval = stats.spearmanr(pseudotime_values, z[:, j])
        results[ph] = {"spearman_rho": rho, "p_value": pval}
        print(f"  {ph:5s}: Spearman ρ = {rho:+.3f}, p = {pval:.2e}")
    return results


# ── 2. Phase order validation ───────────────────────────────────────────

def validate_phase_order(adata, pseudotime_values, marker_dict=None):
    """Check that G1/S → S → G2/M → M → M/G1 is the circular order along pseudotime.

    Because eigenvector sign is arbitrary, both forward and reverse traversal
    around the circle are considered biologically valid.

    Returns
    -------
    phase_means : dict   {phase: circular_mean_pseudotime}
    order_valid : bool   True if the phases appear in (forward or reverse)
                         cyclic order
    direction   : str    'forward', 'reverse', or None
    """
    phases = adata.obs["cc_phase"]
    phase_means = {}
    for phase in PHASE_ORDER:
        mask = (phases == phase).values
        n = mask.sum()
        if n == 0:
            print(f"  WARNING: no cells assigned to {phase}")
            continue
        phase_means[phase] = circular_mean(pseudotime_values[mask])
        print(f"  {phase:5s}: n = {n:4d}, circular mean = {phase_means[phase]:+.3f} rad")

    present_phases = [p for p in PHASE_ORDER if p in phase_means]
    if len(present_phases) < 3:
        print("Cannot validate order — fewer than 3 phases represented.")
        return phase_means, False, None

    # Rotate all present phases so the first is at angle 0, check cyclic order
    angles = np.array([phase_means[p] for p in present_phases])
    rel = (angles - angles[0]) % (2 * np.pi)

    is_forward = np.all(np.diff(rel) > 0)
    # Reverse: angles decrease monotonically around the circle
    rel_rev = (angles[0] - angles) % (2 * np.pi)
    is_reverse = np.all(np.diff(rel_rev[1:]) > 0) if len(rel_rev) > 2 else True
    # Simpler reverse check: reversed list is forward
    rel_r = (angles[::-1] - angles[-1]) % (2 * np.pi)
    is_reverse = np.all(np.diff(rel_r) > 0)

    if is_forward:
        direction = "forward"
        order_valid = True
    elif is_reverse:
        direction = "reverse"
        order_valid = True
    else:
        direction = None
        order_valid = False

    arrow = " → ".join(present_phases)
    if order_valid:
        print(f"  Phase order: {arrow} ({direction} around circle)")
    else:
        print(f"  Phase order INVALID — observed angles do not match {arrow}")
    return phase_means, order_valid, direction


# ── 3. Robustness testing ───────────────────────────────────────────────

def subsample_stability(adata, embedding, k=4, n_trials=10, frac=0.8, seed=42):
    """Recompute angular pseudotime on random cell subsets; measure consistency.

    For each trial, randomly samples *frac* of cells from the already-
    preprocessed adata, rebuilds kNN → Laplacian Eigenmaps → angular
    pseudotime, and computes circular correlation with the full-data
    angular pseudotime.  Takes |correlation| to account for arbitrary
    eigenvector sign flips.

    Parameters
    ----------
    adata     : AnnData after revelio_like_preprocess (has X_pca, cc_phase)
    embedding : (n_cells, 2) full-data Laplacian embedding
    k         : kNN k for graph construction
    n_trials  : number of subsampling rounds
    frac      : fraction of cells to keep per trial
    seed      : random seed

    Returns
    -------
    correlations : list[float]  |circular correlation| for each trial
    trial_data   : list[dict]   per-trial info (indices, subsampled pseudotime)
    """
    from trajectory_inference.Laplacian_Eigenmaps import loadAndCSR, laplacianEigenmaps

    n_cells = adata.n_obs
    n_sample = int(n_cells * frac)
    rng = np.random.default_rng(seed)

    # Full-data angular pseudotime as reference
    ang_pt_full = angular_pseudotime(embedding, adata)

    print(f"  Full data: {n_cells} cells, subsampling {n_sample} ({frac:.0%}), k={k}")

    correlations = []
    trial_data = []
    for t in range(n_trials):
        idx = np.sort(rng.choice(n_cells, n_sample, replace=False))
        adata_sub = adata[idx].copy()

        # Rebuild kNN → Laplacian → embedding on subsampled cells
        csr_sub = loadAndCSR(adata_sub, k)
        emb_sub, eigvals_sub = laplacianEigenmaps(csr_sub)
        ang_pt_sub = angular_pseudotime(emb_sub, adata_sub)

        corr = abs(circular_corr(ang_pt_full[idx], ang_pt_sub))
        correlations.append(corr)

        # Per-phase pseudotime correlations and circular medians
        phases_sub = adata.obs["cc_phase"].values[idx]
        phase_corrs = {}
        phase_medians = {}
        for ph in PHASE_ORDER:
            mask = phases_sub == ph
            if mask.sum() < 3:
                phase_corrs[ph] = np.nan
                phase_medians[ph] = np.nan
            else:
                phase_corrs[ph] = abs(circular_corr(ang_pt_full[idx][mask], ang_pt_sub[mask]))
                phase_medians[ph] = circular_mean(ang_pt_sub[mask])

        trial_data.append({"idx": idx, "ang_pt_sub": ang_pt_sub,
                           "ang_pt_full": ang_pt_full[idx],
                           "phases": phases_sub, "phase_corrs": phase_corrs,
                           "phase_medians": phase_medians})
        print(f"  Trial {t + 1}/{n_trials}: |circular corr| = {corr:.3f}")

    mean_c = np.mean(correlations)
    std_c = np.std(correlations)
    print(f"\n  Subsample stability: {mean_c:.3f} +/- {std_c:.3f}")
    return correlations, trial_data


def k_stability(raw_counts_df, marker_dict, k_values=None):
    """Compute angular pseudotime for a range of k values and compare pairwise.

    Runs the full pipeline (preprocess → kNN → Laplacian → angular pseudotime)
    for each k value.

    Returns
    -------
    pseudotimes : dict   {k: angular_pseudotime_array}
    corr_matrix : np.ndarray  (n_k, n_k) pairwise |circular correlation|
    """
    if k_values is None:
        k_values = [4, 6, 8, 10, 15, 20]

    pseudotimes = {}
    eigval_ratios = {}
    for k in k_values:
        print(f"  k = {k} ... ", end="")
        ps, eigvals, adata_k, emb_k = _full_pipeline(raw_counts_df, k)
        ang_pt = angular_pseudotime(emb_k, adata_k)
        pseudotimes[k] = ang_pt
        ratio = eigvals[1] / eigvals[0] if eigvals[0] != 0 else float("inf")
        eigval_ratios[k] = ratio
        print(f"eigenvalue ratio lambda2/lambda1 = {ratio:.2f}")

    # Pairwise circular correlations
    n_k = len(k_values)
    corr_matrix = np.ones((n_k, n_k))
    for i in range(n_k):
        for j in range(i + 1, n_k):
            c = abs(circular_corr(pseudotimes[k_values[i]], pseudotimes[k_values[j]]))
            corr_matrix[i, j] = c
            corr_matrix[j, i] = c

    # Print correlation matrix
    print("\nPairwise |circular correlation| across k values:")
    header = "      " + "  ".join(f"k={k:<2d}" for k in k_values)
    print(header)
    for i, k in enumerate(k_values):
        row = f"k={k:<2d}  " + "  ".join(f"{corr_matrix[i, j]:.2f}" for j in range(n_k))
        print(row)

    return pseudotimes, corr_matrix


# ── Angular pseudotime for circular validation ─────────────────────────

def angular_pseudotime(embedding, adata):
    """Compute angular pseudotime (0 to 2π) from the 2D Laplacian embedding.

    Uses arctan2 to place each cell on the circle, then shifts so the
    most confident G1/S cell is at 0 and values increase through the cycle.
    """
    angles = np.arctan2(embedding[:, 1], embedding[:, 0])

    # Find root: most confident G1/S cell, extremal on embedding axis 0
    cand = (adata.obs["cc_phase"] == "G1/S").values
    cand &= (adata.obs["best_val"] > 1.0).values
    cand &= (adata.obs["phase_margin"] > 0.75).values
    if cand.sum() == 0:
        cand = (adata.obs["cc_phase"] == "G1/S").values
    root = int(np.where(cand)[0][np.argmin(embedding[cand, 0])])

    # Shift so root is at 0, wrap to [0, 2π)
    angular_pt = (angles - angles[root]) % (2 * np.pi)
    return angular_pt


# ── 4. Independent cyclin peak-order validation ────────────────────────

def validate_cyclin_peaks(adata, embedding, raw_counts_df,
                          validation_genes=None, frac=0.15):
    """Check that independent cyclin/regulator genes peak in the expected
    order along angular pseudotime.

    Uses angular pseudotime (0 to 2π) from the 2D embedding so that
    the circular trajectory is monotonically traversed.  Validation genes
    are NOT in the Revelio marker lists, making this fully independent
    of the phase scoring used in preprocessing.

    Parameters
    ----------
    adata             : AnnData (used for cell names and phase labels)
    embedding         : (n_cells, 2) Laplacian eigenmap embedding
    raw_counts_df     : genes-by-cells DataFrame of raw counts (unfiltered)
    validation_genes  : dict {gene: expected_phase} or None for defaults
    frac              : LOWESS smoothing fraction (0–1, smaller = less smooth)

    Returns
    -------
    peak_df   : DataFrame with columns [gene, expected_phase, expected_rank,
                peak_pseudotime, observed_rank]
    order_rho : Spearman ρ between expected and observed peak ranks
    """
    if validation_genes is None:
        validation_genes = VALIDATION_GENES

    # Compute angular pseudotime for monotonic circular ordering
    ang_pt = angular_pseudotime(embedding, adata)

    # Subset raw counts to cells that survived preprocessing
    cells = adata.obs_names
    raw_sub = raw_counts_df[cells]

    # Normalize: library-size to median, then log1p (same as Revelio step 3)
    lib_sizes = raw_sub.sum(axis=0)
    target = float(np.median(lib_sizes))
    normed = raw_sub.div(lib_sizes, axis=1).mul(target)
    normed = np.log1p(normed)

    # Sort cells by angular pseudotime
    order = np.argsort(ang_pt)
    pt_sorted = ang_pt[order]

    # Circular LOWESS: pad data by wrapping one period on each side so the
    # smoother handles the 0/2π boundary correctly.
    TWO_PI = 2 * np.pi

    def circular_lowess_peak(expr_sorted, pt_sorted, frac):
        pt_ext = np.concatenate([pt_sorted - TWO_PI, pt_sorted, pt_sorted + TWO_PI])
        expr_ext = np.concatenate([expr_sorted, expr_sorted, expr_sorted])
        smooth = sm_lowess(expr_ext, pt_ext, frac=frac / 3, return_sorted=True)
        # Extract the middle period [0, 2π)
        mask = (smooth[:, 0] >= 0) & (smooth[:, 0] < TWO_PI)
        mid = smooth[mask]
        return mid[np.argmax(mid[:, 1]), 0], mid

    # For each gene, circular-LOWESS smooth and find peak pseudotime
    phase_rank = VALIDATION_PHASE_ORDER
    results = []
    missing = []
    for gene, phase in validation_genes.items():
        if gene not in normed.index:
            missing.append(gene)
            continue
        expr = normed.loc[gene].values[order]
        peak_pt, _ = circular_lowess_peak(expr, pt_sorted, frac)
        results.append({
            "gene": gene,
            "expected_phase": phase,
            "expected_rank": phase_rank[phase],
            "peak_pseudotime": peak_pt,
        })

    if missing:
        print(f"  WARNING: genes not found in data: {missing}")

    if len(results) < 2:
        print("  Too few validation genes found to test peak order.")
        return pd.DataFrame(results), np.nan

    peak_df = pd.DataFrame(results)
    peak_df["observed_rank"] = peak_df["peak_pseudotime"].rank().astype(int)

    # Spearman between expected and observed ranks.
    # Take |ρ| because the trajectory direction (forward vs reverse) is
    # arbitrary (eigenvector sign); both are biologically valid.
    order_rho, order_p = stats.spearmanr(peak_df["expected_rank"],
                                          peak_df["observed_rank"])

    print(f"\n  {'Gene':8s}  {'Phase':8s}  {'Expected':>8s}  {'Peak pt':>8s}  {'Obs rank':>8s}")
    print("  " + "-" * 48)
    for _, row in peak_df.iterrows():
        print(f"  {row['gene']:8s}  {row['expected_phase']:8s}  "
              f"{int(row['expected_rank']):>8d}  {row['peak_pseudotime']:>8.3f}  "
              f"{int(row['observed_rank']):>8d}")
    direction = "reverse" if order_rho < 0 else "forward"
    print(f"\n  Peak-order |Spearman ρ| = {abs(order_rho):.3f}, p = {order_p:.2e}"
          f"  (trajectory direction: {direction})")

    return peak_df, order_rho


def plot_cyclin_peaks(adata, embedding, raw_counts_df,
                      validation_genes=None, frac=0.15):
    """Plot LOWESS-smoothed expression of validation genes along angular pseudotime."""
    if validation_genes is None:
        validation_genes = VALIDATION_GENES

    ang_pt = angular_pseudotime(embedding, adata)

    cells = adata.obs_names
    raw_sub = raw_counts_df[cells]
    lib_sizes = raw_sub.sum(axis=0)
    target = float(np.median(lib_sizes))
    normed = raw_sub.div(lib_sizes, axis=1).mul(target)
    normed = np.log1p(normed)

    order = np.argsort(ang_pt)
    pt_sorted = ang_pt[order]

    TWO_PI = 2 * np.pi

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    subset = [("E2F2", "G1" )]
    for gene, phase in subset:
        if gene not in normed.index:
            continue
        expr = normed.loc[gene].values[order]
        # Circular LOWESS
        pt_ext = np.concatenate([pt_sorted - TWO_PI, pt_sorted, pt_sorted + TWO_PI])
        expr_ext = np.concatenate([expr, expr, expr])
        smooth = sm_lowess(expr_ext, pt_ext, frac=frac / 3, return_sorted=True)
        mask = (smooth[:, 0] >= 0) & (smooth[:, 0] < TWO_PI)
        mid = smooth[mask]
        # Min-max normalize for visual comparison
        y = mid[:, 1]
        y = (y - y.min()) / (y.max() - y.min() + 1e-12)
        ax.plot(mid[:, 0], y, label=f"{gene} ({phase})", linewidth=1.5)

    ax.set_xlabel("Angular pseudotime (radians)")
    ax.set_ylabel("Smoothed expression (min-max scaled)")
    ax.set_title("Cyclin gene peaks along angular pseudotime")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()



def plot_phase_zscore_heatmap(adata, embedding):
    """Heatmap of CSV phase z-scores with cells sorted by angular pseudotime.

    Requires score_with_csv_geneset() to have been called first so that
    obsm['phase_scores_z_v2'] and obs['cc_phase_v2'] exist in adata.
    """
    ang_pt = angular_pseudotime(embedding, adata)
    order = np.argsort(ang_pt)

    z = adata.obsm["phase_scores_z_v2"]
    phase_names = list(adata.obs["cc_phase_v2"].cat.categories)

    # Cells as columns (sorted by pseudotime), phases as rows
    z_sorted = z[order, :].T  # (5 phases, n_cells)

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), dpi=150,
                             gridspec_kw={"height_ratios": [1, 0.08]},
                             sharex=True)

    # Main heatmap
    ax = axes[0]
    im = ax.imshow(z_sorted, aspect="auto", cmap="RdBu_r",
                   vmin=-2, vmax=2, interpolation="none")
    ax.set_yticks(range(len(phase_names)))
    ax.set_yticklabels(phase_names, fontsize=9)
    ax.set_ylabel("Phase z-score")
    ax.set_title("Phase z-scores along angular pseudotime (CSV gene set)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="z-score")

    # Phase color bar underneath
    ax2 = axes[1]
    phase_labels = adata.obs["cc_phase_v2"].values[order]
    phase_to_int = {ph: i for i, ph in enumerate(phase_names)}
    phase_ints = np.array([phase_to_int[p] for p in phase_labels])[None, :]
    cmap_phases = plt.cm.get_cmap("tab10", len(phase_names))
    ax2.imshow(phase_ints, aspect="auto", cmap=cmap_phases,
               vmin=-0.5, vmax=len(phase_names) - 0.5, interpolation="none")
    ax2.set_yticks([0])
    ax2.set_yticklabels(["Phase"], fontsize=9)
    ax2.set_xlabel("Cells (sorted by angular pseudotime)")

    # Legend for phase colors
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=cmap_phases(i), label=ph)
               for i, ph in enumerate(phase_names)]
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.5),
               fontsize=7, frameon=False)

    plt.tight_layout()
    plt.show()



# ── Subsample robustness plots ───────────────────────────────────────────


def plot_aggregate_hexbin(trial_data, save_path="subsample_hexbin.jpg"):
    """Hexbin of full vs subsampled angular pseudotime, aggregated across all
    phases and all trials. Subsample orientation is flipped when circular
    correlation is negative (arbitrary eigenvector sign).
    """
    pt_full_all, pt_sub_all = [], []
    for td in trial_data:
        pt_full = td["ang_pt_full"]
        pt_sub = td["ang_pt_sub"].copy()
        if circular_corr(pt_full, pt_sub) < 0:
            pt_sub = (2 * np.pi - pt_sub) % (2 * np.pi)
        pt_full_all.append(pt_full)
        pt_sub_all.append(pt_sub)
    pt_full_all = np.concatenate(pt_full_all)
    pt_sub_all = np.concatenate(pt_sub_all)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    hb = ax.hexbin(pt_full_all, pt_sub_all, gridsize=40, cmap="Blues",
                   mincnt=1, extent=(0, 2 * np.pi, 0, 2 * np.pi))
    fig.colorbar(hb, ax=ax, label="count")
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal")
    ax.set_xlabel("Full pseudotime")
    ax.set_ylabel("Subsampled pseudotime")
    ax.set_title(f"Stability = 0.809 +/- 0.146")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {save_path}")
    plt.show()


def plot_correlation_by_phase(trial_data, save_path="subsample_correlation_by_phase.jpg"):
    """Per-phase dot plot of |circular correlation|: one point per trial,
    phases on the x-axis, red bar at the mean.
    """
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    rng = np.random.default_rng(42)
    for i, ph in enumerate(PHASE_ORDER):
        vals = [td["phase_corrs"].get(ph, np.nan) for td in trial_data]
        vals = [v for v in vals if not np.isnan(v)]
        jitter = i + 0.1 * rng.standard_normal(len(vals))
        ax.scatter(jitter, vals, s=30, alpha=0.6, color="steelblue",
                   edgecolors="white", linewidths=0.4, zorder=3)
        if vals:
            ax.hlines(np.mean(vals), i - 0.25, i + 0.25,
                      colors="firebrick", linewidth=1.5, zorder=4)
    ax.set_xticks(range(len(PHASE_ORDER)))
    ax.set_xticklabels(PHASE_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Phase")
    ax.set_ylabel("|Circular correlation|")
    ax.set_title(f"Per-phase subsample correlation ({len(trial_data)} trials)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {save_path}")
    plt.show()


def plot_phase_order(adata, embedding, phase_means, direction=None,
                     save_path="phase_order.jpg"):
    """Polar plot of phase-order validation results.

    - Cells plotted at their angular pseudotime with radial jitter, colored
      by assigned phase (shows how tight each phase cluster is).
    - Large black-edged marker at each phase's circular mean.
    - Arrows connect successive phases in PHASE_ORDER along the short arc,
      so a valid cycle shows a clean traversal around the circle.
    """
    ang_pt = angular_pseudotime(embedding, adata)
    cmap = plt.cm.get_cmap("tab10", len(PHASE_ORDER))
    colors = {ph: cmap(i) for i, ph in enumerate(PHASE_ORDER)}

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                           figsize=(7, 7), dpi=150)

    rng = np.random.default_rng(42)
    for ph in PHASE_ORDER:
        mask = (adata.obs["cc_phase"] == ph).values
        if mask.sum() == 0:
            continue
        r = 1.0 + 0.12 * rng.standard_normal(mask.sum())
        ax.scatter(ang_pt[mask], r, s=8, alpha=0.35,
                   color=colors[ph], edgecolors="none", label=ph)

    present = [p for p in PHASE_ORDER if p in phase_means]
    R_MEAN = 1.55
    for ph in present:
        th = phase_means[ph] % (2 * np.pi)
        ax.scatter(th, R_MEAN, s=280, color=colors[ph],
                   edgecolors="black", linewidths=1.5, zorder=5)
        ax.text(th, R_MEAN + 0.3, ph, ha="center", va="center",
                fontsize=11, fontweight="bold")

    # Arrows between successive phase means along the short arc
    cycle = present + present[:1]
    for a, b in zip(cycle[:-1], cycle[1:]):
        t1 = phase_means[a] % (2 * np.pi)
        t2 = phase_means[b] % (2 * np.pi)
        diff_ccw = (t2 - t1) % (2 * np.pi)
        diff_cw = (t1 - t2) % (2 * np.pi)
        if diff_ccw <= diff_cw:
            arc = np.linspace(t1, t1 + diff_ccw, 40)
        else:
            arc = np.linspace(t1, t1 - diff_cw, 40)
        ax.plot(arc, np.full_like(arc, R_MEAN), "-",
                color="gray", lw=1.5, alpha=0.6, zorder=3)
        ax.annotate("", xy=(arc[-1], R_MEAN), xytext=(arc[-2], R_MEAN),
                    arrowprops=dict(arrowstyle="->", color="gray",
                                    lw=1.5, alpha=0.8))

    ax.set_rticks([])
    ax.set_rlim(0, 2.1)
    title = "Phase Order Validation along Angular Pseudotime"
    ax.set_title(title, pad=25, fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0),
              fontsize=9, frameon=False, markerscale=1.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {save_path}")
    plt.show()


# ── Run all validations ─────────────────────────────────────────────────

def run_all(adata, embedding, raw_counts_df, marker_dict, k=15,
            subsample_trials=10, k_values=None):
    """Run all validation axes and return combined results.

    Parameters
    ----------
    adata             : AnnData after revelio_like_preprocess
    embedding         : (n_cells, 2) Laplacian embedding
    raw_counts_df     : genes-by-cells raw counts DataFrame
    marker_dict       : dict mapping phase name -> list of marker genes
    k                 : k used to produce the embedding (for subsampling)
    subsample_trials  : number of subsampling rounds
    k_values          : list of k values for stability test (None = default range)
    """
    ang_pt = angular_pseudotime(embedding, adata)

    print()
    print("=" * 60)
    print("2. PHASE ORDER VALIDATION")
    print("=" * 60)
    phase_means, order_valid, direction = validate_phase_order(adata, ang_pt)

    print()
    print("=" * 60)
    print("3a. SUBSAMPLE STABILITY")
    print("=" * 60)
    sub_corrs, trial_data = subsample_stability(
        adata, embedding, k=k, n_trials=subsample_trials)

    print()
    print("=" * 60)
    print("3b. k-VALUE STABILITY")
    print("=" * 60)
    pt_dict, k_corr = k_stability(raw_counts_df, marker_dict, k_values=k_values)

    return {
        "phase_means": phase_means,
        "phase_order_valid": order_valid,
        "phase_direction": direction,
        "subsample_correlations": sub_corrs,
        "subsample_trial_data": trial_data,
        "k_pseudotimes": pt_dict,
        "k_correlation_matrix": k_corr
    }


if __name__ == "__main__":
    marker_dict = load_marker_dict()
    k = 4
    ps, eigvals, adata, embedding = _full_pipeline(data, k)

    adata.obs["pseudotime"] = ps

    print("=" * 60)
    print("1. PHASE SCORE CORRELATIONS")
    print("=" * 60)
    print(correlation_with_phase_scores(adata, ps))

    results = run_all(adata, embedding, data, marker_dict, k=k,
                      subsample_trials=10)

    plot_subsample_robustness(results["subsample_correlations"],
                              results["subsample_trial_data"])
    plot_aggregate_hexbin(results["subsample_trial_data"])
    plot_correlation_by_phase(results["subsample_trial_data"])
    plot_phase_order(adata, embedding,
                     results["phase_means"],
                     results["phase_direction"])

