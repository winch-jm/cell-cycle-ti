import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np


def loadData(filepath):
    # loading in raw data
    wt_data = pd.read_csv(filepath, sep = "\t", index_col = 0)
    return wt_data

############## Original Simple Preprocess ###########################

def simplePreprocess(raw_data_filePath):
    # log1p
    sc.pp.log1p(adata)

    # Highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()

    # PCA
    sc.tl.pca(adata, n_comps=50)

    print(adata)
    print("PCA shape:", adata.obsm['X_pca'].shape)
    adata.write_h5ad("../../data/wt_data.h5ad")

    return adata

################ "Revelio" style Preprocess #########################

# func --> read in gene marker data set and make marker dict for revelio preprocesses

def readMarkerSets(geneSetsFilePath, format, sheet = None):

    if format == "csv":
        geneData = pd.read_csv(geneSetsFilePath)
        dropCols = [c for c in geneData.columns if c.startswith("Unnamed")]
        if len(dropCols) > 0:
            geneData.drop(dropCols, axis = 1, inplace = True)

        markerDict = {}
        for cc in geneData.columns:
            markerDict[cc] = list(geneData[cc].dropna().apply(lambda x: x.strip()).values)

    elif format == "Excel":
        geneData = pd.read_excel(geneSetsFilePath)
        geneSets = geneData.parse(sheet)
        markerDict = {}
        for cc in geneData.columns:
            markerDict[cc] = list(geneSets[cc].dropna().apply(lambda x: x.strip()).values)

    return markerDict

def revelio_like_preprocess(
    counts_df,
    marker_dict,
    min_cells_per_gene=5,
    min_genes_per_cell=1200,
    max_umi_per_cell=10**7,
    min_phase_top_z=1.0,
    min_phase_margin=0,
    min_mean=0.2,
    max_mean=4,
    min_disp=0.5,
    n_pcs=50,
):
    """
    counts_df: genes x cells raw UMI counts (DataFrame)
    marker_dict: ordered dict-like mapping phase -> list of marker genes
    """

    # ----------------------------
    # 1) Create AnnData
    # ----------------------------
    adata = ad.AnnData(X=counts_df.T.values.astype(float))
    adata.obs_names = counts_df.columns.astype(str)
    adata.var_names = counts_df.index.astype(str)

    # ----------------------------
    # 2) QC filtering
    # ----------------------------
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    adata.obs["nUMI"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["nGene"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

    keep = (adata.obs["nGene"] > min_genes_per_cell) & (adata.obs["nUMI"] < max_umi_per_cell)
    adata = adata[keep].copy()

    # ----------------------------
    # 3) Normalize to median UMI, then log1p
    # ----------------------------
    target_sum = float(np.median(np.asarray(adata.X.sum(axis=1)).ravel()))
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Keep a copy of normalized/log data
    adata.layers["log_of_fractions"] = adata.X.copy()


    # ----------------------------
    # 4) Phase scoring
    # ----------------------------
    phase_names = list(marker_dict.keys())

    marker_dict_present = {
        ph: [g for g in genes if g in adata.var_names]
        for ph, genes in marker_dict.items()
    }

    # optional sanity check
    for ph, genes in marker_dict_present.items():
        print(ph, len(genes), genes[:10])

    score_mat = np.zeros((adata.n_obs, len(phase_names)), dtype=float)

    for j, ph in enumerate(phase_names):
        genes = marker_dict_present[ph]
        if len(genes) == 0:
            continue
        idx = adata.var_names.get_indexer(genes)
        score_mat[:, j] = np.asarray(adata[:, idx].X.mean(axis=1)).ravel()

    phase_scores = pd.DataFrame(score_mat, index=adata.obs_names, columns=phase_names)

    # column-wise z-score, then row-wise z-score
    arr = phase_scores.values
    arr = (arr - arr.mean(axis=0)) / arr.std(axis=0, ddof=0)
    arr = (arr - arr.mean(axis=1, keepdims=True)) / arr.std(axis=1, keepdims=True, ddof=0)
    arr = np.nan_to_num(arr)

    phase_scores_z = pd.DataFrame(arr, index=phase_scores.index, columns=phase_scores.columns)

    # ----------------------------
    # 5) Extract best phase / confidence
    # ----------------------------
    best_idx = arr.argmax(axis=1)
    best_val = arr[np.arange(arr.shape[0]), best_idx]

    arr2 = arr.copy()
    arr2[np.arange(arr.shape[0]), best_idx] = -np.inf
    second_idx = arr2.argmax(axis=1)
    second_val = arr2[np.arange(arr2.shape[0]), second_idx]

    phase_margin = best_val - second_val

    adata.obs["cc_phase"] = pd.Categorical(
        [phase_names[i] for i in best_idx],
        categories=phase_names,
        ordered=True
    )
    adata.obs["best_val"] = best_val
    adata.obs["second_val"] = second_val
    adata.obs["phase_margin"] = phase_margin

    adata.obsm["phase_scores"] = phase_scores.values
    adata.obsm["phase_scores_z"] = phase_scores_z.values

    # ----------------------------
    # 6) Filter to confident cycling cells
    # ----------------------------
    keep = (adata.obs["best_val"] > min_phase_top_z) & (adata.obs["phase_margin"] > min_phase_margin)
    adata = adata[keep].copy()

    # ----------------------------
    # 7) HVG selection
    # ----------------------------
    sc.pp.highly_variable_genes(
        adata,
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

    # ----------------------------
    # 8) Scale genes, then PCA
    # ----------------------------
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=n_pcs)

    return adata
