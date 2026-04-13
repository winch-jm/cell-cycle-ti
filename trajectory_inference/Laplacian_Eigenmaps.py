# Implementation of Laplacian eigenmaps to derive circular pseudotime
# goal -- cells that are in similar phase of cell cycle will be closer in 2d embedding

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import anndata as ad
import scanpy as sc
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "openpyxl")

# importing functions for kNN

import sys
sys.path.append('/Users/sophia/Desktop/Fundamentals_of_Bioinformatics/BioInfoFinalProject')
from data_structure.weighted_knn import DenseRows, weighted_knn

# loading in processed dataset
adata = ad.read_h5ad("../data/wt_data.h5ad")

# loading in phase marker data sets
geneData = pd.ExcelFile("../data/GSE142277/gene_sets_GSE142277.xlsx")
phaseList = ["G1/S", "S", "G2/M", "M", "M/G1"]
geneSets = geneData.parse("Gene Sets Used in Analysis")

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

def scoreCells(adata, geneSet = geneSets, phases = phaseList):

    """Input parameters:
        AnnData object from preprocesing
        marker gene set data from study's excel file
        list of phases in gene set data

        Scores each cell based on gene expression of each phase marker gene set and assigns
        highest scoring phase as cell cycle phase for each cell

        Output:
        AnnData object with columns for score of each phase and final phase assignment for each cell"""

    # looping through each phase to score each cell against phase marker gene set
    for phase in phases:
        genes = geneSet[phase].dropna().str.strip().tolist()
        # filtering to genes existing in dataset
        genesPresent = [g for g in genes if g in adata.var_names]
        # scanpy's gene scoring function - computes score of every cell for each phase and saves into new phase col in adata
        sc.tl.score_genes(adata, gene_list=genesPresent, score_name=f"score_{phase}")

    # list of phase col headers
    scoreCols = [f"score_{phase}" for phase in phases]
    # getting highest scored phase for each cell(phase assignment per cell) and ordering based on biological ordering of phases
    adata.obs["phase"] = pd.Categorical(adata.obs[scoreCols].idxmax(axis=1).str.replace("score_", "", regex=False), categories=phases, ordered=True)

    return adata

def findRootCell(adataWithPhase):

    """Input parameters:
    AnnData object from preprocesing and cell cycle phase scoring

    Finds root cell by filtering out cell with highest G1/S score

    Output:
    root cell index in adata"""

    # filtering out G1/S phase cells
    g1Mask = adataWithPhase.obs["phase"] == "G1/S"
    g1Scores = adataWithPhase.obs.loc[g1Mask, "score_G1/S"]
    # root is cell with highest score for G1/S phase
    rootIndex = int(g1Scores.argmax())
    return rootIndex

def pseudotime(embedding, rootIndex = None):

    """Input parameters:
    2D embedding

    If root cell not indicated, converts linear coordinates of 2D embedding into
    angular coordinates. If root cell index indicated, rotates circular manifold so root cell
    is at pseudotime = 0 and other angles are relative to it

    Output:
    pseudotime
    """

    # converts embedding from 2D coordinates to angular coordinates
    pseudo = np.arctan2(embedding[:, 1], embedding[:, 0])

    if rootIndex is not None:
        offset = pseudo[rootIndex]
        pseudo = (pseudo - offset + np.pi) % (2 * np.pi) - np.pi

    return pseudo

csr = loadAndCSR(adata, 15)
embedding, eigvals = laplacianEigenmaps(csr)
scoredAdata = scoreCells(adata, geneSet = geneSets, phases = phaseList)
rootCell = findRootCell(scoredAdata)
ps = pseudotime(embedding, rootIndex = rootCell)

### sanity checks

# should be (1029 - samples, 2 - embeddings)
print("Embedding shape", embedding.shape)
# should be from -pi to pi
print("Pseudotime range", ps.min(), ps.max())
# should be small pos nums
print("Eigenvalues:", eigvals)

def fullLaplacian(adata, k):

    """Input parameters:
    AnnData object from preprocessing and k number of clusters for kNN

    Constructs cell-cell similarity graph, computes graph Laplacian and extracts 2D
    embedding. Scores cells based on gene expression profile for phase marker gene sets
    and assigns cell with phase based on highest score. Finds root cell and calculates
    pseudotime

    Output:
    embedding, eigvals, pseudo"""

    csr = loadAndCSR(adata, k)
    embedding, eigvals = laplacianEigenmaps(csr)
    scoredAdata = scoreCells(adata, geneSet=geneSets, phases=phaseList)
    rootCell = findRootCell(scoredAdata)
    ps = pseudotime(embedding, rootIndex=rootCell)

    return embedding, eigvals, ps