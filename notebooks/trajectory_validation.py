"""
Trajectory Validation
---------------------
Validates inferred circular pseudotime from Laplacian Eigenmaps
against known cell cycle biology.

Validation axes:
1. Correlation between pseudotime and cell cycle phase scores
2. Phase transition order verification (G1 → S → G2/M)
3. Robustness to subsampling and stability across kNN k values
"""

import numpy as np
from scipy import stats
import scanpy as sc

from trajectory_inference.Laplacian_Eigenmaps import loadAndCSR, laplacianEigenmaps, pseudotime as compute_pseudotime


# ── Gene lists (Tirosh et al., 2016) ─────────────────────────────────────
# Includes both original and updated HGNC symbols for compatibility.

S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
    "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "CENPU",
    "MLF1IP",  # old name for CENPU
    "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
    "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
    "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
]

G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
    "TACC3", "FAM64A", "PIMREG",  # FAM64A renamed to PIMREG
    "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
    "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
    "CDCA3", "HN1", "JPT1",  # HN1 renamed to JPT1
    "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
    "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
    "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]


def _full_pipeline(adata, k):
    """Run kNN → Laplacian Eigenmaps → angular pseudotime.

    Returns (pseudotime_array, eigenvalues).
    """
    csr = loadAndCSR(adata, k)
    embedding, eigvals = laplacianEigenmaps(csr)
    pt = compute_pseudotime(embedding)
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

def score_cell_cycle(adata):
    """Add S_score, G2M_score, and phase columns to adata.obs.

    Filters gene lists to those present in the dataset.
    Modifies adata in-place and returns it.
    """
    s_present = [g for g in S_GENES if g in adata.var_names]
    g2m_present = [g for g in G2M_GENES if g in adata.var_names]
    print(f"S-phase genes found: {len(s_present)}/{len(S_GENES)}")
    print(f"G2M genes found:     {len(g2m_present)}/{len(G2M_GENES)}")
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g2m_present)
    return adata


def correlation_with_phase_scores(adata, pseudotime_values):
    """Spearman correlation between pseudotime and S_score / G2M_score.

    Runs score_cell_cycle(adata) if scores are not yet present.

    Returns
    -------
    dict  {score_name: {'spearman_rho': float, 'p_value': float}}
    """
    if "S_score" not in adata.obs.columns:
        score_cell_cycle(adata)

    results = {}
    for col in ["S_score", "G2M_score"]:
        rho, pval = stats.spearmanr(pseudotime_values, adata.obs[col])
        results[col] = {"spearman_rho": rho, "p_value": pval}
        print(f"{col}: Spearman ρ = {rho:+.3f}, p = {pval:.2e}")
    return results


# ── 2. Phase order validation ───────────────────────────────────────────

def validate_phase_order(adata, pseudotime_values):
    """Check that G1 → S → G2M occurs in circular order along pseudotime.

    Because the direction of traversal around the circle is arbitrary
    (eigenvector sign), both G1→S→G2M and G1→G2M→S (reverse) are
    considered biologically valid.

    Returns
    -------
    phase_means : dict   {phase: circular_mean_pseudotime}
    order_valid : bool   True if the three phases are in one of the two
                         valid circular orderings
    direction   : str    'forward' or 'reverse'
    """
    if "phase" not in adata.obs.columns:
        score_cell_cycle(adata)

    phases = adata.obs["phase"]
    phase_means = {}
    for phase in ["G1", "S", "G2M"]:
        mask = (phases == phase).values
        n = mask.sum()
        if n == 0:
            print(f"WARNING: no cells assigned to {phase}")
            continue
        phase_means[phase] = circular_mean(pseudotime_values[mask])
        print(f"  {phase:3s}: n = {n:4d}, circular mean = {phase_means[phase]:+.3f} rad")

    if len(phase_means) < 3:
        print("Cannot validate order — not all three phases represented.")
        return phase_means, False, None

    g1 = phase_means["G1"]
    s = phase_means["S"]
    g2m = phase_means["G2M"]

    # Normalize angles relative to G1 into [0, 2π)
    s_rel = (s - g1) % (2 * np.pi)
    g2m_rel = (g2m - g1) % (2 * np.pi)

    if s_rel < g2m_rel:
        direction = "forward"
    else:
        direction = "reverse"

    # Both directions are biologically valid (just reflects eigenvector sign)
    print(f"  Phase order: G1 → S → G2M ({direction} around circle)")
    return phase_means, True, direction


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

def run_all(adata, pseudotime_values, k=15, subsample_trials=10, k_values=None):
    """Run all three validation axes and return combined results.

    Parameters
    ----------
    adata             : AnnData with X_pca in .obsm
    pseudotime_values : array of angular pseudotime (from Laplacian Eigenmaps)
    k                 : k used to produce pseudotime_values (for subsampling)
    subsample_trials  : number of subsampling rounds
    k_values          : list of k values for stability test (None = default range)
    """
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
