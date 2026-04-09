# Implementation of Laplacian eigenmaps to derive circular pseudotime
# goal -- cells that are in similar phase of cell cycle will be closer in 2d embedding

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# importing processed data and functions for kNN
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_structure.weighted_knn import  DenseRows, weighted_knn

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
    # requesting "SM" for smallest eigvas/vecs and k = 3 to get smallest 2 non-trivial outputs (smallest eigval will be trivial = 0)
    eigvals, eigvecs = eigsh(L, k = nComponents + 1, M = D, which = "SM", tol = 1e-6)

    # sorting eigenvalues in ascending order
    ordered = np.argsort(eigvals)
    eigvals = eigvals[ordered]
    eigvecs = eigvecs[:,ordered]

    # dropping trivial eigenvector
    embedding = eigvecs[:, 1:]
    eigvals = eigvals[1:]

    return embedding, eigvals

def pseudotime(embedding):

    # converts embedding from 2D coordinates to angular coordinates
    pseudo = np.arctan2(embedding[:, 1], embedding[:, 0])

    return pseudo

if __name__ == "__main__":
    from data.preprocess.preprocess import adata

    csr = loadAndCSR(adata, 15)
    embedding, eigvals = laplacianEigenmaps(csr)
    pseudotimeCord = pseudotime(embedding)

    ### sanity checks

    # should be (1029 - samples, 2 - embeddings)
    print("Embedding shape", embedding.shape)
    # should be from -pi to pi
    print("Pseudotime range", pseudotimeCord.min(), pseudotimeCord.max())
    # should be small pos nums
    print("Eigenvalues:", eigvals)