# Diffusion Maps Implementations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_structure.weighted_knn import DenseRows, weighted_knn
from scipy.sparse import diags
import numpy as np
from scipy.sparse.linalg import eigsh  # for symmetric conjugate matrix S


#### making ktilde
def kTilde(adata, k):

    X = adata.obsm["X_pca"]

    N, d = X.shape
    K = np.zeros((N,N))

    # building kNN graph to get neighbor indices
    dr = DenseRows(n=N, d=d, data=X.flatten())
    csr = weighted_knn(dr, k=k)

    # extracting (N x k) indices and Euclidean distances arrays from CSR
    # weighten_knn.py CSR may have >k neighbors per cell due to symmetrization, so sorting by distance and taking k closest neighbors
    indices = np.zeros((N, k), dtype=int)
    distances = np.zeros((N, k))
    for i in range(N):
        nbrs = np.array(csr.indices[csr.indptr[i]:csr.indptr[i + 1]])
        dists = np.linalg.norm(X[nbrs] - X[i], axis=1)
        order = np.argsort(dists)[:k]
        indices[i] = nbrs[order]
        distances[i] = dists[order]

    # adaptive heat kernel
    alpha = 1.0
    for i in range(len(indices)):
        for j in range(k):
            K[i,indices[i,j]] = np.exp(-distances[i,j]**2 /(distances[i,k-1]*distances[indices[i,j],k-1])) # adaptive kernel

    K = (K+K.T)/2 # symmetrize

    # density normalization
    K_tilde = K.copy()
    q = np.sum(K_tilde,axis=1)
    scale = q ** (-alpha)
    K_tilde = (scale[:, None] * K) * scale[None, :]

    return K_tilde

#### getting embedding from ktilde
def diffusion_map_from_Ktilde(K_tilde, n_components=30, t=1, eps=1e-12):
    """
    K_tilde: symmetric sparse matrix after density normalization, before row-normalization
    Returns:
        emb:    (N, n_components) diffusion embedding
        lambdas:(n_components,) nontrivial eigenvalues
        psis:   (N, n_components) right eigenvectors of P
    """
    d = np.asarray(K_tilde.sum(axis=1)).ravel()
    d = np.maximum(d, eps)

    # symmetric conjugate S = D^{-1/2} K_tilde D^{-1/2}
    inv_sqrt_d = 1.0 / np.sqrt(d)
    D_inv_sqrt = diags(inv_sqrt_d)
    S = D_inv_sqrt @ K_tilde @ D_inv_sqrt

    # top eigenpairs
    vals, vecs = eigsh(S, k=n_components + 1, which="LA")
    order = np.argsort(-vals)
    vals = vals[order]
    vecs = vecs[:, order]

    # drop trivial first eigenpair
    lambdas = vals[1:n_components + 1]
    u = vecs[:, 1:n_components + 1]

    # right eigenvectors of P
    psis = u * inv_sqrt_d[:, None]

    # diffusion coordinates at time t
    emb = psis * (lambdas ** t)

    return emb, lambdas, psis

##### finding pseudotime
def diffusion_pseudotime(psis, lambdas, root=0, eps=1e-12):
    """
    psis:    (N, n_components)
    lambdas: (n_components,)
    root:    root cell index

    Returns:
        dpt: (N,) pseudotime/diffusion distance from root
    """
    denom = np.maximum(1.0 - lambdas, eps)  # avoid division by zero
    diff = psis - psis[root, :]             # subtract root from every cell
    dpt_sq = np.sum((diff ** 2) / denom[None, :], axis=1)
    dpt = np.sqrt(dpt_sq)
    return dpt

#### finding root cell
def findRootCell(adata, emb):

    phase0 = "G1.S"

    cand = adata.obs["cc_phase"] == phase0
    cand &= adata.obs["best_val"] > 1.0
    cand &= adata.obs["phase_margin"] > 0.75

    candidate_idx = np.where(cand)[0]

    root = candidate_idx[np.argmin(emb[candidate_idx, 0])]

    return root

#### full implementation of diffusion maps from adata
def fullDiffusion(adata, k):

    kt = kTilde(adata, k)
    emb, lambdas, psis = diffusion_map_from_Ktilde(kt)
    root = findRootCell(adata, emb)
    dpt = diffusion_pseudotime(psis, lambdas, root=root)

    return emb, lambdas, psis, dpt




