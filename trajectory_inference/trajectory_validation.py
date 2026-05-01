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

from trajectory_inference.Laplacian_Eigenmaps import fullLaplacian, loadAndCSR, laplacianEigenmaps, rankPseudo
from trajectory_inference.diffusion_maps import fullDiffusion
from data.preprocess.preprocess import revelio_like_preprocess


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
#data = pd.read_csv("../data/GSE142277/GSM4224316_out_gene_exon_tagged.dge_exonssf002_KO.txt", sep = "\t", index_col = 0)
# Default location of the marker gene set (revelio_gene_sets.csv)
DEFAULT_GENE_SET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "GSE142277", "revelio_gene_sets.csv"
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


def _full_pipeline(data, k, n_components):
    """Run kNN → Laplacian Eigenmaps → pseudotime.

    Returns (pseudotime_array, eigenvalues, adata, embedding).
    """
    adata = revelio_like_preprocess(data, marker_dict)
    embedding, eigvals, ps = fullLaplacian(adata, k, n_components)
    return ps, eigvals, adata, embedding

def geodesic_pipeline(data, k, num_components):
    adata = revelio_like_preprocess(data, marker_dict)
    csr = loadAndCSR(adata, k)
    embedding, eigvals = laplacianEigenmaps(csr, num_components)
    return embedding, eigvals
def full_diffusion_pipeline(data, k, alpha, num_components):
    adata = revelio_like_preprocess(data, marker_dict)
    emb, lambdas, psis, dpt = fullDiffusion(adata, k, alpha, num_components)
    return emb, lambdas, psis, dpt, adata



# ── Circular statistics ──────────────────────────────────────────────────
import numpy as np
TWOPI = 2 * np. pi
def wrap_angle(x):
    return np.mod(x, TWOPI)
def circ_mean (alpha) :
    return np.arctan2(np.sin(alpha).sum(), np.cos(alpha).sum())
def mean_cosine_agreement (thetal, theta2) :
# both assumed already wrapped
    return np.mean(np.cos(thetal - theta2))
def circular_mean(angles):
    """Circular mean of angles in radians."""
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
def align_circular_pseudotime(theta_full, theta_sub, allow_reflection=True):
    theta_full = wrap_angle(np.asarray (theta_full))
    theta_sub = wrap_angle(np.asarray (theta_sub))
    signs = [1, -1] if allow_reflection else [1]
    best = None
    for s in signs:
        theta_trial = wrap_angle(s * theta_sub)
        # optimal rotation = circular mean of angle differences
        diffs = wrap_angle(theta_full - theta_trial)
        delta = circ_mean (diffs)
        aligned = wrap_angle(theta_trial + delta)
        score = mean_cosine_agreement (theta_full, aligned)
        candidate = {
        "aligned": aligned,
        "delta": delta,
        "sign": 5,
        "score": score,
        }
        if best is None or candidate ["score"] > best ["score"]:
            best = candidate
    return best

def aligned_circular_agreement(theta1, theta2):
   
    theta1 = np.asarray(theta1)
    theta2 = np.asarray(theta2)
    if len(theta1) == 0:
        return 0.0
    # Best forward rotation
    alpha_fwd = np.arctan2(np.sum(np.sin(theta1 - theta2)),
                           np.sum(np.cos(theta1 - theta2)))
    fwd = np.mean(np.cos(theta1 - theta2 - alpha_fwd))
    # Best reflected rotation (θ2 → -θ2)
    alpha_rev = np.arctan2(np.sum(np.sin(theta1 + theta2)),
                           np.sum(np.cos(theta1 + theta2)))
    rev = np.mean(np.cos(theta1 + theta2 - alpha_rev))
    return float(max(fwd, rev))
    
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

def generate_subsample_indices(n_cells, n_trials=10, frac=0.8, seed=42):
    """Pre-generate shared subsample indices so Laplacian and diffusion
    stability tests score the exact same cell subsets.

    Returns
    -------
    list[np.ndarray]  sorted integer index arrays, one per trial
    """
    n_sample = int(n_cells * frac)
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_cells, n_sample, replace=False))
            for _ in range(n_trials)]


def subsample_stability(adata, embedding, method,  k=4, n_trials=10, frac=0.8, seed=42,
                         indices=None, n_components=2):
    """Recompute angular pseudotime on random cell subsets; measure consistency.

    For each trial, randomly samples *frac* of cells from the already-
    preprocessed adata, rebuilds kNN → Laplacian Eigenmaps → angular
    pseudotime, and computes aligned circular agreement (rotation- and
    reflection-invariant) with the full-data angular pseudotime.  The
    reflection invariance handles arbitrary eigenvector sign flips.

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
    correlations : list[float]  aligned circular agreement for each trial
    trial_data   : list[dict]   per-trial info (indices, subsampled pseudotime)
    """
    from trajectory_inference.Laplacian_Eigenmaps import loadAndCSR, laplacianEigenmaps

    n_cells = adata.n_obs
    if indices is None:
        indices = generate_subsample_indices(n_cells, n_trials, frac, seed)
    n_trials = len(indices)
    n_sample = len(indices[0])

    # Full-data reference pseudotime, dispatched by method
    if method == "angular":
        pt_full = angular_pseudotime(embedding, adata)
        agreement = aligned_circular_agreement
        phase_summary = circular_mean
    elif method == "geodesic":
        pt_full, _ = signed_geodesic_pseudotime(
            adata, embedding, k=k, n_components=n_components)
        agreement = aligned_circular_agreement
        phase_summary = circular_mean
    else:
        raise ValueError(f"method must be 'angular' or 'geodesic', got {method!r}")

    print(f"  Full data: {n_cells} cells, subsampling {n_sample} ({n_sample/n_cells:.0%}), k={k}")

    correlations = []
    trial_data = []
    for t, idx in enumerate(indices):
        adata_sub = adata[idx].copy()

        # Rebuild kNN → Laplacian → embedding on subsampled cells
        csr_sub = loadAndCSR(adata_sub, k)
        emb_sub, eigvals_sub = laplacianEigenmaps(csr_sub, n_components)

        if method == "angular":
            pt_sub = angular_pseudotime(emb_sub, adata_sub)
        else:  # geodesic
            pt_sub, _ = signed_geodesic_pseudotime(
                adata_sub, emb_sub, k=k, n_components=n_components)

        corr = agreement(pt_full[idx], pt_sub)
        correlations.append(corr)

        # Per-phase pseudotime agreement and per-phase summary
        phases_sub = adata.obs["cc_phase"].values[idx]
        phase_corrs = {}
        phase_medians = {}
        for ph in PHASE_ORDER:
            mask = phases_sub == ph
            if mask.sum() < 3:
                phase_corrs[ph] = np.nan
                phase_medians[ph] = np.nan
            else:
                phase_corrs[ph] = agreement(pt_full[idx][mask], pt_sub[mask])
                phase_medians[ph] = phase_summary(pt_sub[mask])

        trial_data.append({"idx": idx, "ang_pt_sub": pt_sub,
                           "ang_pt_full": pt_full[idx],
                           "phases": phases_sub, "phase_corrs": phase_corrs,
                           "phase_medians": phase_medians})
        print(f"  Trial {t + 1}/{n_trials}: agreement = {corr:.3f}")

    mean_c = np.mean(correlations)
    std_c = np.std(correlations)
    print(f"\n  Subsample stability: {mean_c:.3f} +/- {std_c:.3f}")
    return correlations, trial_data


def subsample_stability_diffusion(adata, embedding, method="angular", k=8,
                                  n_trials=10, frac=0.8, seed=42, indices=None,
                                  n_components=2):
    """Recompute diffusion-maps pseudotime on random cell subsets; measure
    consistency with the full-data result.

    method : "angular"  → arctan2 on first 2 diffusion coords
             "geodesic" → signed_geodesic_pseudotime on the diffusion embedding

    Both branches return values on [0, 2π), so aligned circular agreement
    is the right metric for either.

    Returns
    -------
    correlations : list[float]  aligned circular agreement per trial
    trial_data   : list[dict]   same schema as subsample_stability, so
                                plot_aggregate_hexbin and
                                plot_correlation_by_phase work as-is.
    """
    from trajectory_inference.diffusion_maps import fullDiffusion

    # Full-data diffusion embedding → pseudotime, dispatched by method.
    emb_full, _, _, _ = fullDiffusion(adata, k)
    if method == "angular":
        pt_full = angular_pseudotime(emb_full, adata)
    elif method == "geodesic":
        pt_full, _ = signed_geodesic_pseudotime(
            adata, emb_full, k=k, n_components=n_components)
    else:
        raise ValueError(f"method must be 'angular' or 'geodesic', got {method!r}")

    n_cells = adata.n_obs
    if indices is None:
        indices = generate_subsample_indices(n_cells, n_trials, frac, seed)
    n_trials = len(indices)
    n_sample = len(indices[0])

    print(f"  Full data: {n_cells} cells, subsampling {n_sample} ({n_sample/n_cells:.0%}), k={k}")

    correlations = []
    trial_data = []
    for t, idx in enumerate(indices):
        adata_sub = adata[idx].copy()
        emb_sub, _, _, _ = fullDiffusion(adata_sub, k)
        if method == "angular":
            pt_sub = angular_pseudotime(emb_sub, adata_sub)
        else:
            pt_sub, _ = signed_geodesic_pseudotime(
                adata_sub, emb_sub, k=k, n_components=n_components)

        corr = aligned_circular_agreement(pt_full[idx], pt_sub)
        correlations.append(corr)

        phases_sub = adata.obs["cc_phase"].values[idx]
        phase_corrs = {}
        phase_medians = {}
        for ph in PHASE_ORDER:
            mask = phases_sub == ph
            if mask.sum() < 3:
                phase_corrs[ph] = np.nan
                phase_medians[ph] = np.nan
            else:
                phase_corrs[ph] = aligned_circular_agreement(
                    pt_full[idx][mask], pt_sub[mask])
                phase_medians[ph] = circular_mean(pt_sub[mask])

        trial_data.append({"idx": idx, "ang_pt_sub": pt_sub,
                           "ang_pt_full": pt_full[idx],
                           "phases": phases_sub, "phase_corrs": phase_corrs,
                           "phase_medians": phase_medians})
        print(f"  Trial {t + 1}/{n_trials}: aligned agreement = {corr:.3f}")

    mean_c = np.mean(correlations)
    std_c = np.std(correlations)
    print(f"\n  Subsample stability (diffusion, {method}): {mean_c:.3f} +/- {std_c:.3f}")
    return correlations, trial_data


def k_stability(raw_counts_df, marker_dict, k_values=None):
    """Compute angular pseudotime for a range of k values and compare pairwise.

    Runs the full pipeline (preprocess → kNN → Laplacian → angular pseudotime)
    for each k value.

    Returns
    -------
    pseudotimes : dict   {k: angular_pseudotime_array}
    corr_matrix : np.ndarray  (n_k, n_k) pairwise aligned circular agreement
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

    # Pairwise aligned circular agreement
    n_k = len(k_values)
    corr_matrix = np.ones((n_k, n_k))
    for i in range(n_k):
        for j in range(i + 1, n_k):
            c = aligned_circular_agreement(
                pseudotimes[k_values[i]], pseudotimes[k_values[j]])
            corr_matrix[i, j] = c
            corr_matrix[j, i] = c

    # Print agreement matrix
    print("\nPairwise aligned circular agreement across k values:")
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


def geodesic_distance_from_root(adata, embedding, k, n_components):
    """Geodesic distance from the G1/S root, walking a kNN graph built on
    the Laplacian embedding.

    Builds a distance-weighted kNN graph on the first ``n_components``
    columns of ``embedding``, picks the root with the same logic as
    ``angular_pseudotime``, and runs Dijkstra. On a circular manifold
    these distances fold around the antipode (cells on either side of
    the ring at equal arc length from the root receive equal distance).

    Parameters
    ----------
    adata        : AnnData with obs['cc_phase', 'best_val', 'phase_margin']
    embedding    : (n_cells, d) Laplacian embedding
    k            : neighbors for kNN graph
    n_components : int or None. Number of leading embedding columns to use
                   for the kNN graph. None (default) uses all columns.

    Returns
    -------
    d_root : (n_cells,) geodesic distance from root
    root   : int, index of the root cell
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import dijkstra

    if n_components is None:
        n_components = embedding.shape[1]
    if n_components < 1 or n_components > embedding.shape[1]:
        raise ValueError(
            f"n_components={n_components} out of range for embedding "
            f"with {embedding.shape[1]} columns")
    emb = embedding[:, :n_components]

    cand = (adata.obs["cc_phase"] == "G1/S").values
    cand &= (adata.obs["best_val"] > 1.0).values
    cand &= (adata.obs["phase_margin"] > 0.75).values
    if cand.sum() == 0:
        cand = (adata.obs["cc_phase"] == "G1/S").values
    root = int(np.where(cand)[0][np.argmin(emb[cand, 0])])

    knn = kneighbors_graph(emb, n_neighbors=k, mode="distance")
    knn = knn.maximum(knn.T)

    d_root = dijkstra(knn, indices=root, directed=False)
    if not np.all(np.isfinite(d_root)):
        raise ValueError(f"kNN graph disconnected at k={k} — increase k")
    return d_root, root


def signed_geodesic_pseudotime(adata, embedding, k, n_components):
    """Signed-arc-length geodesic pseudotime on [0, 2π).

    Magnitude is the Dijkstra distance from the G1/S root on a kNN graph
    built over the first ``n_components`` columns of the LE embedding.
    Sign comes from each cell's angular coordinate around the embedding
    centroid (first two LE axes) relative to the root: CCW arc gets +1,
    CW arc gets -1. The angle is used only to pick a side, not as the
    pseudotime magnitude — that still comes from the graph geodesic.

    The signed arc length is rescaled so the antipode lands at ±π and
    the root at 0, then wrapped to [0, 2π). Output is on S¹ and is
    plug-compatible with aligned_circular_agreement, validate_phase_order,
    plot_phase_order, and the angular branch of subsample_stability.
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import dijkstra

    if n_components is None:
        n_components = embedding.shape[1]
    if n_components < 1 or n_components > embedding.shape[1]:
        raise ValueError(
            f"n_components={n_components} out of range for embedding "
            f"with {embedding.shape[1]} columns")
    emb = embedding[:, :n_components]

    cand = (adata.obs["cc_phase"] == "G1/S").values
    cand &= (adata.obs["best_val"] > 1.0).values
    cand &= (adata.obs["phase_margin"] > 0.75).values
    if cand.sum() == 0:
        cand = (adata.obs["cc_phase"] == "G1/S").values
    root = int(np.where(cand)[0][np.argmin(emb[cand, 0])])

    knn = kneighbors_graph(emb, n_neighbors=k, mode="distance")
    knn = knn.maximum(knn.T)

    d_root = dijkstra(knn, indices=root, directed=False)
    if not np.all(np.isfinite(d_root)):
        raise ValueError(f"kNN graph disconnected at k={k} — increase k")

    # Sign from LE angular coordinate around the embedding centroid (first
    # two LE axes). Angle decides the arc; d_root supplies the arc length.
    center = embedding[:, :2].mean(axis=0)
    theta = np.arctan2(embedding[:, 1] - center[1],
                       embedding[:, 0] - center[0])
    delta = (theta - theta[root] + np.pi) % (2 * np.pi) - np.pi
    sign = np.where(delta >= 0, 1.0, -1.0)

    s = sign * d_root
    s_max = float(np.max(np.abs(s)))
    if s_max == 0:
        return np.zeros_like(d_root), root
    geo_pt = (np.pi * s / s_max + 2 * np.pi) % (2 * np.pi)
    return geo_pt, root



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


def plot_aggregate_hexbin(trial_data, save_path="subsample_hexbin.jpg",
                          correlations=None, method_name=None,
                          ax=None, vmin=1, vmax=None, add_colorbar=True):
    """Hexbin of full vs subsampled pseudotime.

    Pass ``ax`` to draw into an existing Axes (panel composition); when
    omitted, creates a standalone fig+ax. Save/show only happen when
    the function created the figure itself. Returns (fig, ax, hexbin)
    so the caller can attach a shared colorbar (use ``add_colorbar=False``
    + ``vmax=shared_max``).
    """
    pt_full_all, pt_sub_all = [], []
    for td in trial_data:
        pt_full = np.asarray(td["ang_pt_full"])
        pt_sub = np.asarray(td["ang_pt_sub"])
        best = align_circular_pseudotime(pt_full, pt_sub, allow_reflection=True)
        pt_full_all.append(pt_full)
        pt_sub_all.append(best["aligned"])
    pt_full_all = np.concatenate(pt_full_all)
    pt_sub_all = np.concatenate(pt_sub_all)
    lo = float(min(pt_full_all.min(), pt_sub_all.min()))
    hi = float(max(pt_full_all.max(), pt_sub_all.max()))
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    else:
        fig = ax.figure

    hb = ax.hexbin(pt_full_all, pt_sub_all, gridsize=40, cmap="Blues",
                   mincnt=1, extent=(lo, hi, lo, hi),
                   vmin=vmin, vmax=vmax)
    if add_colorbar:
        fig.colorbar(hb, ax=ax, label="count")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Full pseudotime")
    ax.set_ylabel("Subsampled pseudotime")

    if correlations is not None and len(correlations) > 0:
        mean_c = float(np.mean(correlations))
        std_c = float(np.std(correlations))
        title = f"Subsample Stability {mean_c:.3f} +/- {std_c:.3f}"
    else:
        title = "Full vs subsampled pseudotime"
    ax.set_title(title)

    if own_fig:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved {save_path}")
        plt.show()
    return fig, ax, hb


def plot_correlation_by_phase(trial_data,
                              save_path="subsample_correlation_by_phase.jpg",
                              ax=None):
    """Per-phase dot plot of aligned circular agreement.

    Pass ``ax`` to draw into an existing Axes; otherwise the function
    creates its own figure. Returns (fig, ax).
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    else:
        fig = ax.figure

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
    ax.set_ylabel("Aligned circular agreement")
    ax.set_title(f"Per-phase subsample agreement ({len(trial_data)} trials)")

    if own_fig:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved {save_path}")
        plt.show()
    return fig, ax


def plot_phase_order(adata, embedding, phase_means, direction=None,
                     save_path="phase_order.jpg", pseudotime=None,
                     title="Phase Order Validation along Angular Pseudotime",
                     ax=None):
    """Polar plot of phase-order validation results.

    Pass ``ax`` to draw into a polar Axes (use add_subplot(..., projection='polar')
    when building the panel figure). Without ``ax`` the function makes its own
    figure. Returns (fig, ax).
    """
    ang_pt = pseudotime if pseudotime is not None else angular_pseudotime(embedding, adata)
    cmap = plt.cm.get_cmap("tab10", len(PHASE_ORDER))
    colors = {ph: cmap(i) for i, ph in enumerate(PHASE_ORDER)}

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                               figsize=(7, 7), dpi=150)
    else:
        fig = ax.figure
        if ax.name != "polar":
            raise ValueError("plot_phase_order requires a polar Axes; "
                             "use projection='polar' when creating it.")

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
    ax.set_title(title, pad=25, fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0),
              fontsize=9, frameon=False, markerscale=1.5)
    if own_fig:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved {save_path}")
        plt.show()
    return fig, ax


def unroll_pseudotime_for_plot(adata, pseudotime):
    """Linearize a circular pseudotime so each phase forms a single mode.

    Two transforms, both invariant w.r.t. the circular structure:
      (1) If S is "behind" G1/S on the circle (eigenvector-sign flip),
          reflect: pt → (2π - pt) % 2π. Restores G1/S → S → G2 → G2/M
          → M/G1 forward direction.
      (2) Find the largest gap between consecutive phase circular means
          and place the [0, 2π) seam in that gap. This guarantees no
          phase straddles the cut, so violins are unimodal.

    Returns the rotated/reflected pseudotime in [0, 2π).
    """
    pt = np.asarray(pseudotime).copy()
    phases = np.asarray(adata.obs["cc_phase"].values)

    def _means(p):
        m = {}
        for ph in PHASE_ORDER:
            mask = phases == ph
            if mask.sum() == 0:
                continue
            m[ph] = circular_mean(p[mask]) % (2 * np.pi)
        return m

    means = _means(pt)

    # (1) Forward direction: S should be just after G1/S on the circle.
    if "G1/S" in means and "S" in means:
        fwd_gap = (means["S"] - means["G1/S"]) % (2 * np.pi)
        if fwd_gap > np.pi:
            pt = (-pt) % (2 * np.pi)
            means = _means(pt)

    # (2) Cut the circle at the largest gap between phase means.
    sorted_means = np.sort(np.array(list(means.values())))
    gaps = np.diff(np.concatenate([sorted_means,
                                   sorted_means[:1] + 2 * np.pi]))
    cut = sorted_means[int(np.argmax(gaps))]
    pt = (pt - cut) % (2 * np.pi)
    return pt


def plot_pseudotime_violin(adata, pseudotime, save_path="pseudotime_violin.jpg",
                           title="Pseudotime distribution by phase",
                           unroll=True, ax=None, show_spearman=True):
    """Violin plot of pseudotime per cell-cycle phase.

    Pass ``ax`` to draw into an existing Axes (panel composition);
    otherwise the function creates its own figure. With ``show_spearman``
    (default True), each y-tick label gets the Spearman ρ between the
    pseudotime and that phase's z-score appended underneath. Returns
    (fig, ax).
    """
    pt = np.asarray(pseudotime).copy().astype(float)
    if unroll:
        pt = unroll_pseudotime_for_plot(adata, pt)
    phases = np.asarray(adata.obs["cc_phase"].values)

    # Per-phase circular alignment with PHASE_ORDER monotonicity:
    #   1. Compute each phase's circular mean.
    #   2. Lift each successive mean by k·2π so it's strictly greater
    #      than the previous phase's lifted mean. This forces violins
    #      onto the axis in canonical phase order regardless of where
    #      circular_mean lands (it can drift to the seam for bimodal
    #      phases like M/G1, which would otherwise dump M/G1 at the
    #      low end).
    #   3. Within each phase, shift cells to lie within ±π of that
    #      lifted target. Cells may end up <0 or >2π — that extends the
    #      linear axis so angular neighbors stay numerical neighbors.
    targets = {}
    prev = None
    for ph in PHASE_ORDER:
        ph_mask = phases == ph
        if ph_mask.sum() == 0:
            continue
        m = circular_mean(pt[ph_mask]) % (2 * np.pi)
        if prev is not None:
            while m <= prev:
                m += 2 * np.pi
        targets[ph] = m
        prev = m

    for ph, target in targets.items():
        ph_mask = phases == ph
        diffs = pt[ph_mask] - target
        aligned = (diffs + np.pi) % (2 * np.pi) - np.pi
        pt[ph_mask] = target + aligned

    mask = np.isin(phases, PHASE_ORDER)
    df = pd.DataFrame({"pseudotime": pt[mask], "phase": phases[mask]})

    cmap = plt.cm.get_cmap("cividis", len(PHASE_ORDER))
    palette = {ph: cmap(i) for i, ph in enumerate(PHASE_ORDER)}

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    else:
        fig = ax.figure
    sns.violinplot(data=df, x="pseudotime", y="phase", order=PHASE_ORDER,
                   palette=palette, inner="box", linewidth=1, ax=ax)

    # Span: from the earliest G1/S cell to the latest M/G1 cell — after
    # per-phase alignment this is the natural cycle range.
    g1s_vals = df.loc[df["phase"] == "G1/S", "pseudotime"]
    mg1_vals = df.loc[df["phase"] == "M/G1", "pseudotime"]
    # if len(g1s_vals) and len(mg1_vals):
    #     ax.set_xlim(float(g1s_vals.min()), float(mg1_vals.max()))

    # x-axis ticks in multiples of π/2.
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    def _pi_formatter(x, _pos):
        n = x / np.pi
        if abs(n) < 1e-9:
            return "0"
        # 3π marks the cyclic wraparound back to G1/S → label as 0
        if abs(n - 3) < 1e-9:
            return "0"
        # whole multiples of π
        if abs(n - round(n)) < 1e-9:
            k = int(round(n))
            if k == 1:
                return "π"
            if k == -1:
                return "−π"
            return f"{k}π"
        # half multiples of π
        n2 = 2 * n
        if abs(n2 - round(n2)) < 1e-9:
            k = int(round(n2))
            if k == 1:
                return "π/2"
            if k == -1:
                return "−π/2"
            return f"{k}π/2"
        return f"{n:.2f}π"

    ax.xaxis.set_major_locator(MultipleLocator(np.pi))
    ax.xaxis.set_major_formatter(FuncFormatter(_pi_formatter))

    # Append per-phase Spearman ρ to the y-tick labels. Use the
    # transformed pt (post-unroll + per-phase alignment) so the numbers
    # match what's visible in the violins. Note: phases whose density
    # peaks in the middle of the axis will get small |ρ| even when the
    # localization is good — Spearman measures monotonicity, not peak
    # alignment. Strong magnitudes are expected only at the cycle ends
    # (G1/S, M/G1).
    if show_spearman and "phase_scores_z" in adata.obsm \
            and "cc_phase" in adata.obs:
        z = adata.obsm["phase_scores_z"]
        phase_names = list(adata.obs["cc_phase"].cat.categories) \
            if hasattr(adata.obs["cc_phase"], "cat") \
            else list(np.unique(adata.obs["cc_phase"]))
        labels = []
        for ph in PHASE_ORDER:
            if ph in phase_names:
                j = phase_names.index(ph)
                rho, _ = stats.spearmanr(pt, z[:, j])
                labels.append(f"{ph}\nρ={rho:+.2f}")
            else:
                labels.append(ph)
        ax.set_yticks(range(len(PHASE_ORDER)))
        ax.set_yticklabels(labels)

    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Cell Cycle Phase")
    ax.set_title(title)
    if own_fig:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved {save_path}")
        plt.show()
    return fig, ax


def panel_figure(image_paths, nrows=2, ncols=2, titles=None,
                 figsize=(15, 8), suptitle=None,
                 save_path="panel.jpg", dpi=600):
    """Combine saved JPG/PNG plots into a single paneled matplotlib figure.

    image_paths : list[str]
        Paths to the images to lay out (in row-major order).
    nrows, ncols : int
        Grid dimensions. nrows*ncols should be >= len(image_paths).
    titles : list[str] | None
        Optional per-panel titles. Length must match image_paths.
    figsize : tuple[float, float]
        Figure size in inches.
    suptitle : str | None
        Optional figure-wide title.
    save_path : str | None
        If given, save the panel figure there.
    dpi : int
        Render resolution.
    """
    import matplotlib.image as mpimg

    paths = list(image_paths)
    n = len(paths)
    if n == 0:
        raise ValueError("image_paths is empty")
    if titles is not None and len(titles) != n:
        raise ValueError("len(titles) must equal len(image_paths)")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True, dpi=dpi)
    axes = np.atleast_2d(axes)
    flat = axes.flatten()

    for i, ax in enumerate(flat):
        if i < n:
            img = mpimg.imread(paths[i])
            ax.imshow(img)
            if titles is not None:
                ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved {save_path}")
    plt.show()


# ── Run all validations ─────────────────────────────────────────────────

def run_all(adata, embedding_ang, embedding_geo, raw_counts_df, marker_dict, k,
            n_components_ang=2, n_components_geo=2,
            subsample_trials=10, k_values=None, subsample_indices=None,
            k_geo=None):
    """Run all validation axes for both pseudotime methods.

    For each method (angular and signed geodesic):
      1. Phase score correlations
      2. Circular trajectory / phase order validation
      3. Subsample stability

    Parameters
    ----------
    adata             : AnnData after revelio_like_preprocess
    embedding_ang     : (n_cells, >=2) embedding for angular pseudotime
    embedding_geo     : (n_cells, n_components_geo) embedding for geodesic
    raw_counts_df     : genes-by-cells raw counts DataFrame
    marker_dict       : dict mapping phase name -> list of marker genes
    k                 : kNN k for the angular subsample rebuild
    k_geo             : kNN k for the geodesic graph (defaults to k)
    n_components_ang  : LE components used for angular trial rebuilds (>=2)
    n_components_geo  : LE components used for geodesic trial rebuilds
    subsample_trials  : number of subsampling rounds
    """
    if k_geo is None:
        k_geo = k

    ang_pt = angular_pseudotime(embedding_ang, adata)
    geo_pt, _ = signed_geodesic_pseudotime(
        adata, embedding_geo, k=k_geo, n_components=n_components_geo)

    # ── Angular ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("ANGULAR — 1. PHASE SCORE CORRELATIONS")
    print("=" * 60)
    corr_ang = correlation_with_phase_scores(adata, ang_pt)

    print()
    print("=" * 60)
    print("ANGULAR — 2. PHASE ORDER VALIDATION")
    print("=" * 60)
    phase_means_ang, order_valid_ang, direction_ang = validate_phase_order(adata, ang_pt)

    print()
    print("=" * 60)
    print("ANGULAR — 3. SUBSAMPLE STABILITY")
    print("=" * 60)
    sub_corrs_ang, trial_data_ang = subsample_stability(
        adata, embedding_ang, "angular", k=k, n_trials=subsample_trials,
        indices=subsample_indices, n_components=n_components_ang)

    # ── Geodesic ────────────────────────────────────────────────────
    # print()
    # print("=" * 60)
    # print("GEODESIC — 1. PHASE SCORE CORRELATIONS")
    # print("=" * 60)
    # corr_geo = correlation_with_phase_scores(adata, geo_pt)

    # print()
    # print("=" * 60)
    # print("GEODESIC — 2. PHASE ORDER VALIDATION")
    # print("=" * 60)
    # phase_means_geo, order_valid_geo, direction_geo = validate_phase_order(adata, geo_pt)

    # print()
    # print("=" * 60)
    # print("GEODESIC — 3. SUBSAMPLE STABILITY")
    # print("=" * 60)
    # sub_corrs_geo, trial_data_geo = subsample_stability(
    #     adata, embedding_geo, "geodesic", k=k_geo, n_trials=subsample_trials,
    #     indices=subsample_indices, n_components=n_components_geo)

    return {
        # angular
        "ang_pt": ang_pt,
        "phase_score_correlations_angular": corr_ang,
        "phase_means_angular": phase_means_ang,
        "phase_order_valid_angular": order_valid_ang,
        "phase_direction_angular": direction_ang,
        "subsample_correlations_angular": sub_corrs_ang,
        "subsample_trial_data_angular": trial_data_ang,
        # geodesic
        # "geo_pt": geo_pt,
        # "phase_score_correlations_geodesic": corr_geo,
        # "phase_means_geodesic": phase_means_geo,
        # "phase_order_valid_geodesic": order_valid_geo,
        # "phase_direction_geodesic": direction_geo,
        # "subsample_correlations_geodesic": sub_corrs_geo,
        # "subsample_trial_data_geodesic": trial_data_geo,
    }


def run_all_diffusion(adata, embedding, k=8, subsample_trials=10,
                      subsample_indices=None, n_components_geo=2):
    """Run all validation axes for both pseudotime methods on the
    diffusion-maps embedding:
      1. Phase score correlations
      2. Circular trajectory / phase order validation
      3. Subsample stability

    Returns a single dict with keys suffixed _angular / _geodesic, mirroring
    run_all().
    """
    ang_pt = angular_pseudotime(embedding, adata)
    # geo_pt, _ = signed_geodesic_pseudotime(
    #     adata, embedding, k=k, n_components=n_components_geo)

    # ── Angular ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("DIFFUSION ANGULAR — 1. PHASE SCORE CORRELATIONS")
    print("=" * 60)
    corr_ang = correlation_with_phase_scores(adata, ang_pt)

    print()
    print("=" * 60)
    print("DIFFUSION ANGULAR — 2. PHASE ORDER VALIDATION")
    print("=" * 60)
    phase_means_ang, order_valid_ang, direction_ang = validate_phase_order(adata, ang_pt)

    print()
    print("=" * 60)
    print("DIFFUSION ANGULAR — 3. SUBSAMPLE STABILITY")
    print("=" * 60)
    sub_corrs_ang, trial_data_ang = subsample_stability_diffusion(
        adata, embedding, method="angular", k=k, n_trials=subsample_trials,
        indices=subsample_indices, n_components=n_components_geo)

    # ── Geodesic ────────────────────────────────────────────────────
    # print()
    # print("=" * 60)
    # print("DIFFUSION GEODESIC — 1. PHASE SCORE CORRELATIONS")
    # print("=" * 60)
    # corr_geo = correlation_with_phase_scores(adata, geo_pt)

    # print()
    # print("=" * 60)
    # print("DIFFUSION GEODESIC — 2. PHASE ORDER VALIDATION")
    # print("=" * 60)
    # phase_means_geo, order_valid_geo, direction_geo = validate_phase_order(adata, geo_pt)

    # print()
    # print("=" * 60)
    # print("DIFFUSION GEODESIC — 3. SUBSAMPLE STABILITY")
    # print("=" * 60)
    # sub_corrs_geo, trial_data_geo = subsample_stability_diffusion(
    #     adata, embedding, method="geodesic", k=k, n_trials=subsample_trials,
    #     indices=subsample_indices, n_components=n_components_geo)

    return {
        # angular
        "ang_pt": ang_pt,
        "phase_score_correlations_angular": corr_ang,
        "phase_means_angular": phase_means_ang,
        "phase_order_valid_angular": order_valid_ang,
        "phase_direction_angular": direction_ang,
        "subsample_correlations_angular": sub_corrs_ang,
        "subsample_trial_data_angular": trial_data_ang,
        # geodesic
        # "geo_pt": geo_pt,
        # "phase_score_correlations_geodesic": corr_geo,
        # "phase_means_geodesic": phase_means_geo,
        # "phase_order_valid_geodesic": order_valid_geo,
        # "phase_direction_geodesic": direction_geo,
        # "subsample_correlations_geodesic": sub_corrs_geo,
        # "subsample_trial_data_geodesic": trial_data_geo,
    }


if __name__ == "__main__":
    
    marker_dict = load_marker_dict()
    #panel_figure(["pseudotime_violin_angular.jpg","pseudotime_violin_diffusion_angular.jpg", "subsample_hexbin_angular.jpg","subsample_hexbin_diffusion_angular.jpg"])
    k = 5
    n_components = 51
    n_trials = 10
    # ps_lap, eigvals_lap, adata_lap, emb_lap = _full_pipeline(data, k, n_components)
    # newAdata = rankPseudo(adata_lap, eigvals_lap, ps_lap)
    # print(adata_lap.obs["cc_phase"].value_counts())

    # cmap = plt.cm.get_cmap("tab10", len(PHASE_ORDER))
    # colors = [cmap(i) for i in range(len(PHASE_ORDER))]

    # fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    # for phase, color in zip(PHASE_ORDER, colors):
    #     mask = newAdata.obs["cc_phase"] == phase
    #     ax.scatter(emb_lap[mask, 1], emb_lap[mask, 0],
    #                label=phase, color=color, s=10)
    # ax.set_title(f"Laplacian Eigenmap Embedding by Cell Cycle Phase (k={k})")
    # ax.set_xlabel("Embedding C2")
    # ax.set_ylabel("Embedding C1")
    # ax.legend(title="Cell Cycle Phase",
    #           bbox_to_anchor=(1.01, 1), loc="upper left")
    # fig.tight_layout()
    # plt.show()

    # ── Laplacian Eigenmaps full validation ──────────────────────────
    print("\n" + "#" * 60)
    print("# LAPLACIAN EIGENMAPS PIPELINE")
    print("#" * 60)

    ps_lap, eigvals_lap, adata_lap, emb_lap = _full_pipeline(data, k, n_components)
    print(adata_lap.obs["cc_phase"].value_counts())
    geo_embedding, geo_eig = geodesic_pipeline(data, k, n_components)

    # Shared subsample indices so both methods evaluate the exact same cell subsets.
    shared_idx = generate_subsample_indices(adata_lap.n_obs, n_trials=n_trials)
    lap_results = run_all(adata_lap, emb_lap, geo_embedding, data, marker_dict, k=k,
                          n_components_ang=2, n_components_geo=n_components,
                          subsample_trials=n_trials,
                          subsample_indices=shared_idx)

    # # ── Plots: angular ────────────────────────────────────────────────
    plot_aggregate_hexbin(
        lap_results["subsample_trial_data_angular"],
        save_path="subsample_hexbin_angular.jpg",
        correlations=lap_results["subsample_correlations_angular"],
        method_name="Angular (Laplacian Eigenmaps)")
    plot_correlation_by_phase(
        lap_results["subsample_trial_data_angular"],
        save_path="subsample_correlation_by_phase_angular.jpg")
    plot_phase_order(
        adata_lap, emb_lap,
        lap_results["phase_means_angular"],
        lap_results["phase_direction_angular"],
        pseudotime=lap_results["ang_pt"],
        save_path="phase_order_angular.jpg",
        title="Phase Order — Angular Pseudotime")
    plot_pseudotime_violin(
        adata_lap, lap_results["ang_pt"],
        save_path="pseudotime_violin_angular.jpg",
        title="Pseudotime by Phase — Laplacian Angular")

    # ── Plots: signed geodesic ────────────────────────────────────────
    # plot_aggregate_hexbin(
    #     lap_results["subsample_trial_data_geodesic"],
    #     save_path="subsample_hexbin_geodesic.jpg",
    #     correlations=lap_results["subsample_correlations_geodesic"],
    #     method_name="Signed Geodesic (Laplacian Eigenmaps)")
    # plot_correlation_by_phase(
    #     lap_results["subsample_trial_data_geodesic"],
    #     save_path="subsample_correlation_by_phase_geodesic.jpg")
    # plot_phase_order(
    #     adata_lap, geo_embedding,
    #     lap_results["phase_means_geodesic"],
    #     lap_results["phase_direction_geodesic"],
    #     pseudotime=lap_results["geo_pt"],
    #     save_path="phase_order_geodesic.jpg",
    #     title="Phase Order — Signed Geodesic Pseudotime")

    # ── Diffusion-maps full validation ─────────────────────────────────
    print("\n" + "#" * 60)
    print("# DIFFUSION MAPS PIPELINE")
    print("#" * 60)
    alpha = 0.824
    k = 8 
    num_components = 10
    emb_diff, lambdas, psis, dpt, adata_diff = full_diffusion_pipeline(data, k, alpha, num_components)
    # assert adata_diff.n_obs == adata_lap.n_obs, \
    #     "adata mismatch — shared indices would be invalid"

    diff_results = run_all_diffusion(adata_diff, emb_diff, k=k,
                                     subsample_trials=n_trials,
                                     subsample_indices=shared_idx,
                                     n_components_geo=n_components)

    # # ── Plots: diffusion angular ──────────────────────────────────────
    plot_aggregate_hexbin(
        diff_results["subsample_trial_data_angular"],
        save_path="subsample_hexbin_diffusion_angular.jpg",
        correlations=diff_results["subsample_correlations_angular"],
        method_name="Angular (Diffusion Maps)")
    plot_correlation_by_phase(
        diff_results["subsample_trial_data_angular"],
        save_path="subsample_correlation_by_phase_diffusion_angular.jpg")
    plot_phase_order(
        adata_diff, emb_diff,
        diff_results["phase_means_angular"],
        diff_results["phase_direction_angular"],
        pseudotime=diff_results["ang_pt"],
        save_path="phase_order_diffusion_angular.jpg",
        title="Phase Order — Diffusion Angular Pseudotime")
    plot_pseudotime_violin(
        adata_diff, diff_results["ang_pt"],
        save_path="pseudotime_violin_diffusion_angular.jpg",
        title="Pseudotime by Phase — Diffusion Angular")

    # # # ── Plots: diffusion signed geodesic ──────────────────────────────
    # # plot_aggregate_hexbin(
    # #     diff_results["subsample_trial_data_geodesic"],
    # #     save_path="subsample_hexbin_diffusion_geodesic.jpg",
    # #     correlations=diff_results["subsample_correlations_geodesic"],
    # #     method_name="Signed Geodesic (Diffusion Maps)")
    # # plot_correlation_by_phase(
    # #     diff_results["subsample_trial_data_geodesic"],
    # #     save_path="subsample_correlation_by_phase_diffusion_geodesic.jpg")
    # # plot_phase_order(
    # #     adata_diff, emb_diff,
    # #     diff_results["phase_means_geodesic"],
    # #     diff_results["phase_direction_geodesic"],
    # #     pseudotime=diff_results["geo_pt"],
    # #     save_path="phase_order_diffusion_geodesic.jpg",
    # #     title="Phase Order — Diffusion Signed Geodesic Pseudotime")
    # # plot_pseudotime_violin(
    # #     adata_diff, diff_results["geo_pt"],
    # #     save_path="pseudotime_violin_diffusion_geodesic.jpg",
    # #     title="Pseudotime by Phase — Diffusion Signed Geodesic")

    # # ── Combined panel: violins on top, hexbins on bottom, shared cbar ──
    fig = plt.figure(figsize=(10, 8), dpi=600)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.04])
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])

    plot_pseudotime_violin(adata_lap, lap_results["ang_pt"], ax=ax_tl,
                           save_path=None,
                           title="Pseudotime by Phase — Laplacian Angular")
    plot_pseudotime_violin(adata_diff, diff_results["ang_pt"], ax=ax_tr,
                           save_path=None,
                           title="Pseudotime by Phase — Diffusion Angular")

    # Match the violin axes box to the square hexbins below.
    ax_tl.set_box_aspect(1)
    ax_tr.set_box_aspect(1)
    # Drop redundant y-axis labels on the right column.
    ax_tr.set_ylabel("")

    _, _, hb_l = plot_aggregate_hexbin(
        lap_results["subsample_trial_data_angular"], ax=ax_bl,
        add_colorbar=False, save_path=None,
        correlations=lap_results["subsample_correlations_angular"])
    _, _, hb_r = plot_aggregate_hexbin(
        diff_results["subsample_trial_data_angular"], ax=ax_br,
        add_colorbar=False, save_path=None,
        correlations=diff_results["subsample_correlations_angular"])
    ax_br.set_ylabel("")

    vmax = float(max(hb_l.get_array().max(), hb_r.get_array().max()))
    hb_l.set_clim(1, vmax)
    hb_r.set_clim(1, vmax)
    fig.colorbar(hb_l, cax=cax, label="Cell count")
    fig.tight_layout()
    fig.savefig("panel_violin_hexbin_shared_cbar.jpg",
                dpi=1200, bbox_inches="tight")
    print("  Saved panel_violin_hexbin_shared_cbar.jpg")
    plt.show()