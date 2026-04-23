import numpy as np
import pandas as pd
import anndata as ad
import sys
sys.path.append('../trajectory_inference/')
from toyDataWithDensity import make_circle_varying_density
from diffusion_maps import kTilde, diffusion_map_from_Ktilde
from Laplacian_Eigenmaps import loadAndCSR, laplacianEigenmaps
from data_structure.weighted_knn import DenseRows, weighted_knn
import scipy.sparse as sp

def make_validation_adata(n_points=500, seed=0):
    # uniform circle, no noise
    df = make_circle_varying_density(
        noise=0.0,
        density_ratio=1.0,
        n_points=n_points,
        seed=seed,
    )

    obs = df[["angle"]].copy()
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    adata = ad.AnnData(X=df[["x", "y"]].to_numpy(), obs=obs)
    adata.obsm["X_pca"] = adata.X.copy()
    return adata


def components_vs_angle_df(angles, emb, method_name, n_plot_components=2):
    """
    angles : array-like shape (n,)
    emb    : array-like shape (n, m)
    """
    angles = np.asarray(angles)
    emb = np.asarray(emb)

    order = np.argsort(angles)
    angles_sorted = angles[order]
    emb_sorted = emb[order]

    rows = []
    for comp_idx in range(min(n_plot_components, emb.shape[1])):
        for a, val in zip(angles_sorted, emb_sorted[:, comp_idx]):
            rows.append({
                "Method": method_name,
                "angle": a,
                "component": comp_idx + 1,
                "value": val,
            })
    return pd.DataFrame(rows)


def eigenvalue_df(eigvals, method_name):
    eigvals = np.asarray(eigvals)
    return pd.DataFrame({
        "Method": method_name,
        "component_rank": np.arange(1, len(eigvals) + 1),
        "eigenvalue": eigvals,
    })


def generate_validation_panel_data(
    n_points=500,
    seed=0,
    k=20,
    alpha=0.5,
    n_components=5,
    symmetrization="union",
    drop_trivial=True
):
    adata = make_validation_adata(n_points=n_points, seed=seed)
    X = adata.obsm["X_pca"]
    N, d = X.shape

    dr = DenseRows(n=N, d=d, data=X.flatten())

    # shared adaptive Gaussian graph
    csr = weighted_knn(
        dr,
        k=k,
        metric="euclidean",
        weighting="adaptive_gaussian",
        symmetrization=symmetrization,
    )
    csr = sp.csr_matrix((csr.data, csr.indices, csr.indptr),shape=(N, N))

    
    # LE
    le_emb, le_eigvals = laplacianEigenmaps(csr, nComponents=n_components,drop_trivial=drop_trivial)

    # DM
    K_tilde = kTilde(csr, alpha=alpha)
    dm_emb, dm_lambdas, dm_psis = diffusion_map_from_Ktilde(
        K_tilde,
        n_components=n_components,
        t=1,
        drop_trivial=drop_trivial
    )

    # store embeddings
    adata.obsm["le_validation"] = le_emb
    adata.obsm["dm_validation"] = dm_emb

    # component-vs-angle dataframes
    le_comp_df = components_vs_angle_df(
        adata.obs["angle"].values,
        le_emb,
        method_name="LE",
        n_plot_components=2,
    )
    dm_comp_df = components_vs_angle_df(
        adata.obs["angle"].values,
        dm_emb,
        method_name="DM",
        n_plot_components=2,
    )
    components_df = pd.concat([le_comp_df, dm_comp_df], ignore_index=True)

    # eigenvalue dataframes
    le_eval_df = eigenvalue_df(le_eigvals, "LE")
    dm_eval_df = eigenvalue_df(dm_lambdas, "DM")
    spectrum_df = pd.concat([le_eval_df, dm_eval_df], ignore_index=True)

    return adata, components_df, spectrum_df