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

from trajectory_inference.Laplacian_Eigenmaps import loadAndCSR, laplacianEigenmaps, pseudotime


# Canonical order of cell cycle phases around the circle
PHASE_ORDER = ["G1/S", "S", "G2/M", "M", "M/G1"]

# Default location of the marker gene spreadsheet (relative to this file)
DEFAULT_GENE_SET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "GSE142277", "gene_sets_GSE142277.xlsx"
)


def load_marker_dict(path=DEFAULT_GENE_SET_PATH, sheet_name="Gene Sets Used in Analysis"):
    """Load the 5-phase marker dictionary from the gene-sets spreadsheet.

    Returns an ordered dict mapping phase name -> list of gene symbols.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    marker_dict = {}
    for col in df.columns:
        marker_dict[col] = list(df[col].dropna().astype(str).str.strip().values)
    return marker_dict


def _full_pipeline(adata, k):
    """Run kNN → Laplacian Eigenmaps → angular pseudotime.

    Returns (pseudotime_array, eigenvalues).
    """
    csr = loadAndCSR(adata, k)
    embedding, eigvals = laplacianEigenmaps(csr)
    pt = pseudotime(embedding)
    return pt, eigvals


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


# ── 1. Correlation with cell cycle phase scores ─────────────────────────

def score_cell_cycle(adata, marker_dict):
    """Revelio-style 5-phase cell cycle scoring with double z-normalization.

    For each phase, computes mean log-normalized expression of marker genes,
    then z-scores column-wise (across cells) and row-wise (across phases).
    The phase with the highest z-score is assigned as ``cc_phase``.

    Looks up marker genes in ``adata.raw`` when available so that scoring
    does not depend on highly-variable-gene filtering; falls back to
    ``adata.X`` otherwise.

    Modifies adata in-place:
        obs['cc_phase']       — categorical phase label
        obs['best_val']       — top z-score per cell (confidence)
        obs['phase_margin']   — gap between top and second-best z-score
        obsm['phase_scores']  — raw mean-expression matrix (n_cells × 5)
        obsm['phase_scores_z']— double z-scored matrix (n_cells × 5)

    Returns the modified adata.
    """
    phase_names = list(marker_dict.keys())

    # Use the full (pre-HVG) expression matrix for marker lookup if available
    if adata.raw is not None:
        var_names = adata.raw.var_names
        X = adata.raw.X
    else:
        var_names = adata.var_names
        X = adata.X

    # Filter each phase's markers to those present in the dataset
    marker_present = {
        ph: [g for g in genes if g in var_names] for ph, genes in marker_dict.items()
    }
    for ph, genes in marker_present.items():
        print(f"  {ph:5s}: {len(genes)}/{len(marker_dict[ph])} markers found")

    # Mean expression per phase
    score_mat = np.zeros((adata.n_obs, len(phase_names)), dtype=float)
    for j, ph in enumerate(phase_names):
        genes = marker_present[ph]
        if len(genes) == 0:
            continue
        idx = var_names.get_indexer(genes)
        score_mat[:, j] = np.asarray(X[:, idx].mean(axis=1)).ravel()

    # Double z-score: column-wise then row-wise
    arr = score_mat.copy()
    arr = (arr - arr.mean(axis=0)) / arr.std(axis=0, ddof=0)
    arr = (arr - arr.mean(axis=1, keepdims=True)) / arr.std(axis=1, keepdims=True, ddof=0)
    arr = np.nan_to_num(arr)

    # Assign best phase + confidence
    best_idx = arr.argmax(axis=1)
    best_val = arr[np.arange(arr.shape[0]), best_idx]

    arr2 = arr.copy()
    arr2[np.arange(arr.shape[0]), best_idx] = -np.inf
    second_val = arr2.max(axis=1)
    phase_margin = best_val - second_val

    adata.obs["cc_phase"] = pd.Categorical(
        [phase_names[i] for i in best_idx],
        categories=phase_names,
        ordered=True,
    )
    adata.obs["best_val"] = best_val
    adata.obs["phase_margin"] = phase_margin
    adata.obsm["phase_scores"] = score_mat
    adata.obsm["phase_scores_z"] = arr

    return adata


def correlation_with_phase_scores(adata, pseudotime_values, marker_dict=None):
    """Spearman correlation between pseudotime and each phase's z-score.

    Runs score_cell_cycle(adata, marker_dict) if scores are not yet present.

    Returns
    -------
    dict  {phase_name: {'spearman_rho': float, 'p_value': float}}
    """
    if "cc_phase" not in adata.obs.columns:
        if marker_dict is None:
            raise ValueError("marker_dict required when scores not yet computed")
        score_cell_cycle(adata, marker_dict)

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
    if "cc_phase" not in adata.obs.columns:
        if marker_dict is None:
            raise ValueError("marker_dict required when scores not yet computed")
        score_cell_cycle(adata, marker_dict)

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

def subsample_stability(adata, k=15, n_trials=10, frac=0.8, seed=42):
    """Recompute pseudotime on random cell subsets; measure consistency.

    For each trial, randomly samples *frac* of cells, reruns the full
    pipeline, and computes circular correlation with the full-data
    pseudotime (on shared cells).  Takes |correlation| to account for
    arbitrary eigenvector sign flips.

    Returns
    -------
    list[float]  absolute circular correlation for each trial
    """
    n_cells = adata.n_obs
    n_sample = int(n_cells * frac)
    rng = np.random.default_rng(seed)

    print(f"Computing full pseudotime (k={k}) ...")
    pt_full, _ = _full_pipeline(adata, k)

    correlations = []
    for t in range(n_trials):
        idx = np.sort(rng.choice(n_cells, n_sample, replace=False))
        adata_sub = adata[idx].copy()

        print(f"  Trial {t + 1}/{n_trials} ({n_sample} cells) ... ", end="")
        pt_sub, _ = _full_pipeline(adata_sub, k)

        corr = abs(circular_corr(pt_full[idx], pt_sub))
        correlations.append(corr)
        print(f"|circular corr| = {corr:.3f}")

    mean_c = np.mean(correlations)
    std_c = np.std(correlations)
    print(f"Subsample stability: {mean_c:.3f} ± {std_c:.3f}")
    return correlations


def k_stability(adata, k_values=None):
    """Compute pseudotime for a range of k values and compare pairwise.

    Returns
    -------
    pseudotimes : dict   {k: pseudotime_array}
    corr_matrix : np.ndarray  (n_k, n_k) pairwise |circular correlation|
    """
    if k_values is None:
        k_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]

    pseudotimes = {}
    eigval_ratios = {}
    for k in k_values:
        print(f"  k = {k} ... ", end="")
        pt, eigvals = _full_pipeline(adata, k)
        pseudotimes[k] = pt
        ratio = eigvals[1] / eigvals[0] if eigvals[0] != 0 else float("inf")
        eigval_ratios[k] = ratio
        print(f"eigenvalue ratio λ2/λ1 = {ratio:.2f}")

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


# ── Run all validations ─────────────────────────────────────────────────

def run_all(adata, pseudotime_values, marker_dict, k=15, subsample_trials=10, k_values=None):
    """Run all three validation axes and return combined results.

    Parameters
    ----------
    adata             : AnnData with X_pca in .obsm
    pseudotime_values : array of angular pseudotime (from Laplacian Eigenmaps)
    marker_dict       : dict mapping phase name -> list of marker genes
    k                 : k used to produce pseudotime_values (for subsampling)
    subsample_trials  : number of subsampling rounds
    k_values          : list of k values for stability test (None = default range)
    """
    # Compute phase scores once, up-front, so downstream steps reuse them
    if "cc_phase" not in adata.obs.columns:
        print("Scoring cell cycle phases (Revelio-style double z-score) ...")
        score_cell_cycle(adata, marker_dict)
        print()

    print("=" * 60)
    print("1. CORRELATION WITH CELL CYCLE PHASE SCORES")
    print("=" * 60)
    corr_results = correlation_with_phase_scores(adata, pseudotime_values)

    print()
    print("=" * 60)
    print("2. PHASE ORDER VALIDATION")
    print("=" * 60)
    phase_means, order_valid, direction = validate_phase_order(adata, pseudotime_values)

    print()
    print("=" * 60)
    print("3a. SUBSAMPLE STABILITY")
    print("=" * 60)
    sub_corrs = subsample_stability(adata, k=k, n_trials=subsample_trials)

    print()
    print("=" * 60)
    print("3b. k-VALUE STABILITY")
    print("=" * 60)
    pt_dict, k_corr = k_stability(adata, k_values=k_values)

    return {
        "phase_score_correlations": corr_results,
        "phase_means": phase_means,
        "phase_order_valid": order_valid,
        "phase_direction": direction,
        "subsample_correlations": sub_corrs,
        "k_pseudotimes": pt_dict,
        "k_correlation_matrix": k_corr,
    }


if __name__ == "__main__":
    from data.preprocess.preprocess import adata

    marker_dict = load_marker_dict()
    k = 15
    pt, _ = _full_pipeline(adata, k)
    run_all(adata, pt, marker_dict, k=k)
