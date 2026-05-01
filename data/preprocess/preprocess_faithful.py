import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


def _zscore(x, axis=0):
    mean = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True)
    std[std == 0] = np.nan
    return np.nan_to_num((x - mean) / std)


def _get_batch_ids(cell_names, dataset_id="data1"):
    """
    Revelio convention:
    if cell IDs contain '_', the prefix before '_' is treated as batch ID.
    Otherwise all cells are assigned to one batch.
    """
    cell_names = pd.Index(cell_names).astype(str)
    has_underscore = cell_names.str.contains("_")

    if has_underscore.any():
        return cell_names.str.split("_").str[0].to_numpy()
    else:
        return np.array([f"{dataset_id}Batch1"] * len(cell_names))


def _marker_dict_to_present(marker_dict, var_names):
    return {
        ph: [g for g in genes if pd.notna(g) and g in var_names]
        for ph, genes in marker_dict.items()
    }


def _prune_markers_by_bucket_correlation(X_log, var_names, marker_dict_present, cor_threshold=0.2):
    """
    Revelio-style step:
    For each marker bucket, compute the average population expression profile
    across all genes in that bucket. Remove genes whose expression profile
    across cells correlates poorly with that bucket-average profile.

    X_log: cells x genes log-of-fractions matrix
    """
    pruned = {}
    gene_phase = pd.Series(index=var_names, dtype="object")

    for phase, genes in marker_dict_present.items():
        if len(genes) == 0:
            pruned[phase] = []
            continue

        idx = pd.Index(var_names).get_indexer(genes)
        idx = idx[idx >= 0]

        if len(idx) == 0:
            pruned[phase] = []
            continue

        bucket_expr = X_log[:, idx]
        bucket_avg = np.asarray(bucket_expr.mean(axis=1)).ravel()

        keep_genes = []
        for g, j in zip(np.array(var_names)[idx], idx):
            gene_expr = np.asarray(X_log[:, j]).ravel()

            if np.std(gene_expr) == 0 or np.std(bucket_avg) == 0:
                cor = np.nan
            else:
                cor = np.corrcoef(gene_expr, bucket_avg)[0, 1]

            if np.isfinite(cor) and cor > cor_threshold:
                keep_genes.append(g)
                gene_phase.loc[g] = phase

        pruned[phase] = keep_genes

    return pruned, gene_phase


def _score_phases_for_cells(X_log, var_names, marker_dict_pruned):
    """
    Compute phase scores as average log-fraction expression over retained marker genes,
    then normalize column-wise and row-wise, matching the Schwabe/Revelio-style
    z-score phase assignment logic.
    """
    phase_names = list(marker_dict_pruned.keys())
    score_mat = np.zeros((X_log.shape[0], len(phase_names)), dtype=float)

    for j, phase in enumerate(phase_names):
        genes = marker_dict_pruned[phase]
        if len(genes) == 0:
            score_mat[:, j] = 0.0
            continue

        idx = pd.Index(var_names).get_indexer(genes)
        idx = idx[idx >= 0]
        score_mat[:, j] = np.asarray(X_log[:, idx].mean(axis=1)).ravel()

    # Revelio-style normalization:
    # first across cells within each phase, then across phases within each cell
    score_z_col = _zscore(score_mat, axis=0)
    score_z = _zscore(score_z_col, axis=1)

    return score_mat, score_z


def revelio_faithful_preprocess(
    counts_df,
    marker_dict,
    dataset_id="data1",
    lower_n_gene_cutoff=500,
    upper_n_umi_cutoff=10**7,
    cc_phase_assign_based_on_individual_batches=True,
    cc_phase_assign_threshold_for_cor_single_gene_to_avg_expression=0.2,
    cc_phase_assign_max_diff_between_highest_scores=1,
    cc_phase_assign_threshold_highest_phase_score=0.75,
    cc_phase_assign_threshold_second_highest_phase_score=0.5,
    min_cells_per_gene_after_filter=5,
    pca_genes="variableGenes",
    min_mean=0.2,
    max_mean=4,
    min_disp=0.5,
    n_pcs=50,
):
    """
    More faithful Python approximation of Schwabe et al. / Revelio preprocessing.

    Parameters
    ----------
    counts_df:
        Raw UMI count matrix, genes x cells.
    marker_dict:
        Ordered mapping from phase name to marker genes.
        Expected order should be cyclic, e.g.
        G1.S, S, G2, G2.M, M.G1.
    """

    phase_names = list(marker_dict.keys())

    # ----------------------------
    # 1) Create AnnData from raw counts
    # ----------------------------
    adata = ad.AnnData(X=counts_df.T.values.astype(float))
    adata.obs_names = counts_df.columns.astype(str)
    adata.var_names = counts_df.index.astype(str)

    # ----------------------------
    # 2) Initial Revelio-style raw-count QC
    # ----------------------------
    adata.obs["nUMI"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["nGene"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

    keep_cells = (
        (adata.obs["nGene"] > lower_n_gene_cutoff)
        & (adata.obs["nUMI"] < upper_n_umi_cutoff)
    )
    adata = adata[keep_cells].copy()

    # Remove genes with zero counts after cell filtering
    keep_genes = np.asarray(adata.X.sum(axis=0)).ravel() > 0
    adata = adata[:, keep_genes].copy()

    adata.obs["nUMI"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["nGene"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    adata.var["nUMI"] = np.asarray(adata.X.sum(axis=0)).ravel()
    adata.var["nCell"] = np.asarray((adata.X > 0).sum(axis=0)).ravel()

    # ----------------------------
    # 3) Median-UMI log-of-fractions normalization
    # ----------------------------
    scaling_factor_umi = float(np.median(adata.obs["nUMI"]))
    sc.pp.normalize_total(adata, target_sum=scaling_factor_umi)
    sc.pp.log1p(adata)

    adata.layers["log_of_fractions"] = adata.X.copy()
    adata.uns["scalingFactorUMI"] = scaling_factor_umi

    # ----------------------------
    # 4) Batch IDs using Revelio convention
    # ----------------------------
    adata.obs["batchID"] = _get_batch_ids(adata.obs_names, dataset_id=dataset_id)

    # ----------------------------
    # 5) Marker pruning + phase scoring
    # ----------------------------
    marker_dict_present = _marker_dict_to_present(marker_dict, adata.var_names)

    all_phase_scores = np.full((adata.n_obs, len(phase_names)), np.nan)
    all_phase_scores_z = np.full((adata.n_obs, len(phase_names)), np.nan)
    gene_phase_all = pd.Series(index=adata.var_names, dtype="object")

    if cc_phase_assign_based_on_individual_batches:
        batch_values = adata.obs["batchID"].unique()
    else:
        batch_values = ["__all__"]

    for batch in batch_values:
        if batch == "__all__":
            cell_mask = np.ones(adata.n_obs, dtype=bool)
        else:
            cell_mask = adata.obs["batchID"].to_numpy() == batch

        X_batch = np.asarray(adata.layers["log_of_fractions"][cell_mask, :])

        marker_pruned, gene_phase = _prune_markers_by_bucket_correlation(
            X_batch,
            adata.var_names,
            marker_dict_present,
            cor_threshold=cc_phase_assign_threshold_for_cor_single_gene_to_avg_expression,
        )

        score_raw, score_z = _score_phases_for_cells(
            X_batch,
            adata.var_names,
            marker_pruned,
        )

        all_phase_scores[cell_mask, :] = score_raw
        all_phase_scores_z[cell_mask, :] = score_z

        gene_phase_all = gene_phase_all.combine_first(gene_phase)

    phase_scores = pd.DataFrame(
        all_phase_scores,
        index=adata.obs_names,
        columns=phase_names,
    )
    phase_scores_z = pd.DataFrame(
        all_phase_scores_z,
        index=adata.obs_names,
        columns=[f"{p}_zScore" for p in phase_names],
    )

    # ----------------------------
    # 6) Highest and second-highest phase scores
    # ----------------------------
    arr = phase_scores_z.to_numpy()

    best_idx = arr.argmax(axis=1)
    best_val = arr[np.arange(arr.shape[0]), best_idx]

    arr2 = arr.copy()
    arr2[np.arange(arr.shape[0]), best_idx] = -np.inf

    second_idx = arr2.argmax(axis=1)
    second_val = arr2[np.arange(arr2.shape[0]), second_idx]

    n_phases = len(phase_names)
    phase_dist = np.abs(best_idx - second_idx)
    phase_dist = np.minimum(phase_dist, n_phases - phase_dist)

    suspected_doublet = (
        (phase_dist > cc_phase_assign_max_diff_between_highest_scores)
        & (second_val > cc_phase_assign_threshold_second_highest_phase_score)
    )

    low_confidence = best_val < cc_phase_assign_threshold_highest_phase_score

    adata.obs["cc_phase"] = pd.Categorical(
        [phase_names[i] for i in best_idx],
        categories=phase_names,
        ordered=True,
    )
    adata.obs["second_cc_phase"] = pd.Categorical(
        [phase_names[i] for i in second_idx],
        categories=phase_names,
        ordered=True,
    )
    adata.obs["highestPhaseScore"] = best_val
    adata.obs["secondHighestPhaseScore"] = second_val
    adata.obs["phaseDistanceTop2"] = phase_dist
    adata.obs["isOutlierSuspectedDoublet"] = suspected_doublet
    adata.obs["isOutlierNoConfidenceInPhaseScore"] = low_confidence

    adata.obsm["phase_scores"] = phase_scores.to_numpy()
    adata.obsm["phase_scores_z"] = phase_scores_z.to_numpy()
    adata.uns["phase_score_columns"] = phase_names
    adata.uns["phase_score_z_columns"] = list(phase_scores_z.columns)

    adata.var["ccPhase"] = gene_phase_all.reindex(adata.var_names).to_numpy()

    # ----------------------------
    # 7) Revelio-style phase-assignment filtering
    # ----------------------------
    keep_cells = ~(suspected_doublet | low_confidence)
    adata = adata[keep_cells].copy()

    # ----------------------------
    # 8) Repeat QC after phase filtering
    # ----------------------------
    adata.obs["nUMI"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["nGene"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

    keep_cells = (
        (adata.obs["nGene"] > lower_n_gene_cutoff)
        & (adata.obs["nUMI"] < upper_n_umi_cutoff)
    )
    adata = adata[keep_cells].copy()

    keep_genes = np.asarray((adata.X > 0).sum(axis=0)).ravel() >= min_cells_per_gene_after_filter
    adata = adata[:, keep_genes].copy()

    # ----------------------------
    # 9) PCA genes
    # ----------------------------
    if pca_genes == "variableGenes":
        sc.pp.highly_variable_genes(
            adata,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
        )
        adata = adata[:, adata.var["highly_variable"]].copy()
    elif pca_genes == "allGenes":
        pass
    else:
        raise ValueError("pca_genes must be either 'variableGenes' or 'allGenes'.")

    # ----------------------------
    # 10) Scale and PCA
    # ----------------------------
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=n_pcs)

    return adata