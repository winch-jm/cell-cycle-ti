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
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_structure.weighted_knn import DenseRows, weighted_knn

####### preprocessed data loading and kNN
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

    early = "G1.S"
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

#adata = revelio_like_preprocess(data, marker_dict)
#embedding, eigvals, ps = fullLaplacian(adata, k = 10)

### sanity checks
#print("Embedding shape", embedding.shape)
#print("Pseudotime range", ps.min(), ps.max())
# should be small pos nums
#print("Eigenvalues:", eigvals)