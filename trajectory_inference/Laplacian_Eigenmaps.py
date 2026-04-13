# Implementation of Laplacian eigenmaps to derive circular pseudotime
# goal -- cells that are in similar phase of cell cycle will be closer in 2d embedding

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import anndata as ad
import scanpy as sc
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "openpyxl")

# importing functions for kNN
import sys
sys.path.append('/Users/sophia/Desktop/Fundamentals_of_Bioinformatics/BioInfoFinalProject')
from data_structure.weighted_knn import DenseRows, weighted_knn

# loading in raw data
data = pd.read_csv("../data/GSE142277/GSM4224315_out_gene_exon_tagged.dge_exonssf002_WT.txt", sep = "\t", index_col = 0)

# loading in phase marker data sets
geneData = pd.ExcelFile("../data/GSE142277/gene_sets_GSE142277.xlsx")
phaseList = ["G1/S", "S", "G2/M", "M", "M/G1"]
geneSets = geneData.parse("Gene Sets Used in Analysis")

# making dictionary of each phase's gene markers
marker_dict = {}
for cc in geneSets.columns:
    marker_dict[cc] = list(geneSets[cc].dropna().apply(lambda x: x.strip()).values)

#### revelio style preprocessing from diffusion maps implementation notebook
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


##### data loading and kNN
def loadAndCSR(adata, k):

    """Input -- AnnData object from data preprocessing

    Extracts PCA embeddings from AnnData object, structures into dense rows
    and builds cell-cell similarity graph in custom CSR format, which is converted
    to scipy sparse csr for Laplacian function input

    Output -- Scipy sparse CSR"""

    pca = adata.obsm["X_pca"]
    n, d = pca.shape
    dr = DenseRows(n = n, d = d, data = pca.flatten())
    customCSR = weighted_knn(dr, k = k)

    # converting custom CSR to scipy sparse CSR
    CSR = sp.csr_matrix((customCSR.data, customCSR.indices, customCSR.indptr),shape=(n, n))

    return CSR

#### laplacian computation
def laplacianEigenmaps(CSR, nComponents = 2):
    """Input Parameters:
    CSR: compressed sparse row matrix of cell-cell similarity graph
    nComponents: number of embedding dimensions = 2 to get 2D embedding coordinates per cell required for circular pseudotime calculation

    Computes graph Laplacian from cell-cell similarity CSR matrix
    and solves eigenproblem to get 2 smallest non-trivial eigenvalues for embeddings

    Output:
    embedding: shape (nCells, nComponents)
    eigenvalues: shape (nComponents)"""

    # degree matrix -- sum of similarities for each cell (degree of each node in graph) along diagonal of sparse diagonal matrix
    degrees = np.array(CSR.sum(axis = 1)).flatten()
    D = sp.diags(degrees, format = 'csr')

    # graph laplacian L = D - W where W = CSR
    L = D - CSR

    # solution to eigenproblem Lv = lamda(Dv)
    # setting a random, seeded v0 for eigsh
    rng = np.random.default_rng(50)
    v0 = rng.random(L.shape[0])
    # requesting "SM" for smallest eigvas/vecs and k = 3 to get smallest 2 non-trivial outputs (smallest eigval will be trivial = 0)
    eigvals, eigvecs = eigsh(L, k = nComponents + 1, M = D, which = "SM", tol = 1e-6, v0=v0)

    # sorting eigenvalues in ascending order
    ordered = np.argsort(eigvals)
    eigvals = eigvals[ordered]
    eigvecs = eigvecs[:,ordered]

    # dropping trivial eigenvector
    embedding = eigvecs[:, 1:]
    eigvals = eigvals[1:]

    return embedding, eigvals

#### Finging root cell
def findRootCell(adata, embedding):

    """Input parameters:
    AnnData object from revelio preprocesing
    embedding

    Finds root cell by finding most confident, early phase (G1/S) cell from preprocessed adata

    Output:
    root cell index in adata"""

    # filtering out best G1/S phase

    early = "G1/S"
    candidate = adata.obs["cc_phase"] == early
    candidate &= adata.obs["best_val"] > 1.0
    candidate &= adata.obs["phase_margin"] > 0.75

    candIndex = np.where(candidate)[0]

    root = int(candIndex[np.argmin(embedding[candIndex, 0])])

    return root


##### pseudotime derivation based on root cell
def laplacianPseudotime(embedding, eigvals, rootIndex=0, eps=1e-12):

    """Input parameters:
    embedding
    eigenvals
    rootIndex -> pseudotime = 0 at root
    eps -> small value to avoid division by zero

    pseudotime = weighted Euclidean distance from root cell to all other cell
    in embedding space where each dimension is scaled by 1/(1 - eigval).
    Dimensions with eigenvalues near 1 are down-weighted

    Output:
    pseudotime for each cell"""

    denom = np.maximum(1.0 - eigvals, eps)
    diff = embedding - embedding[rootIndex, :]
    dpt_sq = np.sum((diff ** 2) / denom[None, :], axis=1)
    pseudo = np.sqrt(dpt_sq)

    return pseudo

#### integration of helper functions for full laplacian eigenmap method
def fullLaplacian(adata, k):

    """Input parameters:
    AnnData object from revelio preprocessing and k number of clusters for kNN

    Constructs cell-cell similarity graph, computes graph Laplacian and extracts 2D
    embedding. Calculates pseudotime for cells as Euclidean distance from root cell.

    Output:
    embedding, eigvals, pseudo"""

    csr = loadAndCSR(adata, k)
    embedding, eigvals = laplacianEigenmaps(csr)
    rootIndex = findRootCell(adata, embedding)
    ps = laplacianPseudotime(embedding, eigvals, rootIndex=rootIndex, eps=1e-12)

    return embedding, eigvals, ps

############### FULL LAPLACIAN IMPLEMENTATION

adata = revelio_like_preprocess(data, marker_dict)
embedding, eigvals, ps = fullLaplacian(adata, k = 10)

### sanity checks
print("Embedding shape", embedding.shape)
print("Pseudotime range", ps.min(), ps.max())
# should be small pos nums
print("Eigenvalues:", eigvals)