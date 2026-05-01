import numpy as np
import pandas as pd
import anndata as ad
import sys
sys.path.append('../trajectory_inference/')
from toyDataWithDensity import make_circle_varying_density
from diffusion_maps import kTilde, diffusion_map_from_Ktilde
from Laplacian_Eigenmaps import loadAndCSR, laplacianEigenmaps
from data_structure.weighted_knn import DenseRows, weighted_knn
from sklearn.datasets import make_swiss_roll
import scipy.sparse as sp

def make_validation_adata(n_points=500, seed=0, type='uniform_circle'):
    # uniform circle, no noise
    if type == 'uniform_circle':
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
    elif type == "swiss_roll":
        coords, position = make_swiss_roll(
            n_samples=n_points,
            noise=0.0,
            random_state=seed,
        )

        df = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
                "position": position,
            }
        )

        obs = df[["position"]].copy()
        obs.index = [f"cell_{i}" for i in range(len(obs))]

        adata = ad.AnnData(
            X=df[["x", "y", "z"]].to_numpy(),
            obs=obs,
        )
        adata.obsm["X_pca"] = adata.X.copy()
    else:
        print('Invalid dataset type: use \'uniform_circle\' or \'swiss_roll\'')
        return None
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


def generate_uniform_circle_validation_panel_data(
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


def components_vs_position_df(position, emb, method_name, n_plot_components=2):
    """
    For Swiss roll validation: compare embedding components against the
    intrinsic unrolled coordinate returned by sklearn.make_swiss_roll.
    """
    position = np.asarray(position)
    emb = np.asarray(emb)

    order = np.argsort(position)
    position_sorted = position[order]
    emb_sorted = emb[order]

    rows = []
    for comp_idx in range(min(n_plot_components, emb.shape[1])):
        for p, val in zip(position_sorted, emb_sorted[:, comp_idx]):
            rows.append({
                "Method": method_name,
                "position": p,
                "component": comp_idx + 1,
                "value": val,
            })

    return pd.DataFrame(rows)

def generate_swiss_roll_validation_panel_data(
    n_points=500,
    seed=0,
    k=20,
    alpha=0.5,
    n_components=5,
    symmetrization="union",
    drop_trivial=True,
    noise=0.0,
    weighting="adaptive_gaussian",
):
    """
    Generate data for a Swiss roll validation panel.

    Returns
    -------
    adata : AnnData
        Contains original Swiss roll coordinates and DM/LE embeddings.
    components_df : pd.DataFrame
        Long-form dataframe of embedding components vs. Swiss roll position.
    spectrum_df : pd.DataFrame
        Eigenvalue/lambda spectrum for LE and DM.
    embedding_df : pd.DataFrame
        Long-form dataframe for plotting 2D embeddings colored by position.
    """

    # Make Swiss roll directly here so n_points, seed, and noise are respected
    coords, position = make_swiss_roll(
        n_samples=n_points,
        noise=noise,
        random_state=seed,
    )

    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "position": position,
        }
    )

    obs = df[["position"]].copy()
    obs.index = [f"cell_{i}" for i in range(n_points)]

    adata = ad.AnnData(
        X=df[["x", "y", "z"]].to_numpy(),
        obs=obs,
    )
    adata.obsm["X_pca"] = adata.X.copy()

    X = adata.obsm["X_pca"]
    N, d = X.shape

    dr = DenseRows(n=N, d=d, data=X.flatten())

    # Shared graph for both methods
    csr = weighted_knn(
        dr,
        k=k,
        metric="euclidean",
        weighting=weighting,
        symmetrization=symmetrization,
    )
    csr = sp.csr_matrix(
        (csr.data, csr.indices, csr.indptr),
        shape=(N, N),
    )

    # Laplacian Eigenmaps
    le_emb, le_eigvals = laplacianEigenmaps(
        csr,
        nComponents=n_components,
        drop_trivial=drop_trivial,
    )

    # Diffusion Maps
    K_tilde = kTilde(csr, alpha=alpha)
    dm_emb, dm_lambdas, dm_psis = diffusion_map_from_Ktilde(
        K_tilde,
        n_components=n_components,
        t=1,
        drop_trivial=drop_trivial,
    )

    adata.obsm["le_validation"] = le_emb
    adata.obsm["dm_validation"] = dm_emb

    # Component vs intrinsic Swiss roll coordinate
    le_comp_df = components_vs_position_df(
        adata.obs["position"].values,
        le_emb,
        method_name="LE",
        n_plot_components=2,
    )

    dm_comp_df = components_vs_position_df(
        adata.obs["position"].values,
        dm_emb,
        method_name="DM",
        n_plot_components=2,
    )

    components_df = pd.concat(
        [le_comp_df, dm_comp_df],
        ignore_index=True,
    )

    # Spectrum dataframe
    spectrum_df = pd.concat(
        [
            eigenvalue_df(le_eigvals, "LE"),
            eigenvalue_df(dm_lambdas, "DM"),
        ],
        ignore_index=True,
    )

    # Embedding dataframe for plotting DM/LE coordinates colored by position
    embedding_df = pd.concat(
        [
            pd.DataFrame({
                "Method": "LE",
                "component_1": le_emb[:, 0],
                "component_2": le_emb[:, 1],
                "position": adata.obs["position"].values,
            }),
            pd.DataFrame({
                "Method": "DM",
                "component_1": dm_emb[:, 0],
                "component_2": dm_emb[:, 1],
                "position": adata.obs["position"].values,
            }),
        ],
        ignore_index=True,
    )

    return adata, components_df, spectrum_df, embedding_df