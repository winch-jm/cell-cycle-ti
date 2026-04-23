# Diffusion Maps Implementations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_structure.weighted_knn import DenseRows, weighted_knn
from scipy.sparse import diags
import numpy as np
from scipy.sparse.linalg import eigsh  # for symmetric conjugate matrix S

import numpy as np
import scipy.sparse as sp

def custom_csr_to_scipy(C):
    """
    Convert custom CSR to scipy.sparse.csr_matrix.
    """
    return sp.csr_matrix(
        (
            np.asarray(C.data, dtype=float),
            np.asarray(C.indices, dtype=int),
            np.asarray(C.indptr, dtype=int),
        ),
        shape=(C.n, C.n),
    )

#### making ktilde

def kTilde(K, alpha=1.0, eps=1e-12):
    """
    Alpha-normalize an affinity matrix K.

    Parameters
    ----------
    K : custom CSR or scipy.sparse matrix
        Symmetric nonnegative affinity matrix.
    alpha : float
        Density-normalization exponent.

    Returns
    -------
    K_tilde : scipy.sparse.csr_matrix
        Alpha-normalized sparse matrix.
    """
    if hasattr(K, "indptr") and hasattr(K, "indices") and hasattr(K, "data") and hasattr(K, "n"):
        K = custom_csr_to_scipy(K)
    elif not sp.issparse(K):
        K = sp.csr_matrix(K)
    else:
        K = K.tocsr()

    q = np.asarray(K.sum(axis=1)).ravel()
    q = np.maximum(q, eps)

    scale = q ** (-alpha)
    D_alpha = diags(scale)

    K_tilde = D_alpha @ K @ D_alpha
    return K_tilde.tocsr()

# def kTilde(K, alpha=1.0):

#     # X = adata.obsm["X_pca"]

#     # N, d = X.shape
#     # K = np.zeros((N,N))

#     # # building kNN graph to get neighbor indices
#     # dr = DenseRows(n=N, d=d, data=X.flatten())
#     # csr = weighted_knn(dr, k=k,metric='euclidean')

#     # # extracting (N x k) indices and Euclidean distances arrays from CSR
#     # # weighten_knn.py CSR may have >k neighbors per cell due to symmetrization, so sorting by distance and taking k closest neighbors
#     # indices = np.zeros((N, k), dtype=int)
#     # distances = np.zeros((N, k))
#     # for i in range(N):
#     #     nbrs = np.array(csr.indices[csr.indptr[i]:csr.indptr[i + 1]])
#     #     dists = np.linalg.norm(X[nbrs] - X[i], axis=1)
#     #     order = np.argsort(dists)[:k]
#     #     indices[i] = nbrs[order]
#     #     distances[i] = dists[order]

#     # # adaptive heat kernel
#     # for i in range(len(indices)):
#     #     for j in range(k):
#     #         K[i,indices[i,j]] = np.exp(-distances[i,j]**2 /(distances[i,k-1]*distances[indices[i,j],k-1])) # adaptive kernel

#     # K = (K+K.T)/2 # symmetrize

#     # density normalization
#     K_tilde = csr.copy()
#     q = np.sum(K_tilde,axis=1)
#     scale = q ** (-alpha)
#     K_tilde = (scale[:, None] * K) * scale[None, :]

#     return K_tilde

#### getting embedding from ktilde
def diffusion_map_from_Ktilde(K_tilde, n_components=30, t=1, eps=1e-12):
    """
    K_tilde: symmetric sparse matrix after alpha-normalization, before row-normalization

    Returns
    -------
    emb : (N, n_components)
        Diffusion embedding
    lambdas : (n_components,)
        Nontrivial eigenvalues
    psis : (N, n_components)
        Right eigenvectors of the Markov matrix P
    """
    if not sp.issparse(K_tilde):
        K_tilde = sp.csr_matrix(K_tilde)
    else:
        K_tilde = K_tilde.tocsr()

    d = np.asarray(K_tilde.sum(axis=1)).ravel()
    d = np.maximum(d, eps)

    inv_sqrt_d = 1.0 / np.sqrt(d)
    D_inv_sqrt = diags(inv_sqrt_d)

    # symmetric conjugate
    S = D_inv_sqrt @ K_tilde @ D_inv_sqrt
    S = S.tocsr()

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
    denom = np.maximum(1.0 - lambdas, eps)
    diff = psis - psis[root, :]
    dpt_sq = np.sum((diff ** 2) / denom[None, :], axis=1)
    return np.sqrt(dpt_sq)

#### finding root cell
def findRootCell(adata, emb):

    phase0 = "G1.S"

    cand = adata.obs["cc_phase"] == phase0
    cand &= adata.obs["best_val"] > 1.0
    cand &= adata.obs["phase_margin"] > 0.75

    candidate_idx = np.where(cand)[0]

    root = candidate_idx[np.argmin(emb[candidate_idx, 0])]

    return root


def fullDiffusion(adata, k, alpha=1.0, n_components=30, t=1,
                  weighting="adaptive_gaussian", symmetrization="union"):
    """
    Full diffusion maps pipeline using shared graph construction.
    Assumes weighted_knn can build the affinity graph you want.
    """
    X = adata.obsm["X_pca"]
    N, d = X.shape

    dr = DenseRows(n=N, d=d, data=X.flatten())

    K = weighted_knn(
        dr,
        k=k,
        metric="euclidean",
        weighting=weighting,
        symmetrization=symmetrization,
    )

    K_tilde = kTilde(K, alpha=alpha)
    emb, lambdas, psis = diffusion_map_from_Ktilde(K_tilde, n_components=n_components, t=t)
    root = findRootCell(adata, emb)
    dpt = diffusion_pseudotime(psis, lambdas, root=root)

    return emb, lambdas, psis, dpt


