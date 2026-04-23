import sys
sys.path.append('../data/toy_datasets/')
from toyDataWithDensity import make_circle_varying_density
import numpy as np
import anndata as ad
import numpy as np
from tqdm import tqdm
import pandas as pd
sys.path.append('../trajectory_inference/')
from diffusion_maps import kTilde, diffusion_map_from_Ktilde
from Laplacian_Eigenmaps import loadAndCSR, laplacianEigenmaps
from trajectory_validation import align_circular_pseudotime,aligned_circular_agreement

def wrap_to_pi(angle):
    """Wrap angles to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def mean_abs_wrapped_error(theta_true, theta_pred):
    """
    Mean absolute circular error between aligned angles.

    Parameters
    ----------
    theta_true : array-like
    theta_pred : array-like

    Returns
    -------
    float
        Mean absolute wrapped angular error (in radians, in [0, pi]).
    """
    theta_true = np.asarray(theta_true)
    theta_pred = np.asarray(theta_pred)

    delta = wrap_to_pi(theta_true - theta_pred)
    return float(np.mean(np.abs(delta)))

def radius_cv(emb):
    """
    Coefficient of variation of radius in 2D embedding.

    Parameters
    ----------
    emb : array-like, shape (n_samples, 2)

    Returns
    -------
    float
        std(radius) / mean(radius)
    """
    emb = np.asarray(emb)
    
    r = np.sqrt(np.sum(emb**2, axis=1))

    # avoid division by zero
    mean_r = np.mean(r)
    if mean_r == 0:
        return 0.0

    return float(np.std(r) / mean_r)


def __main__():
    noise_levels = np.arange(0,0.51,0.05)
    density_ratios = np.arange(1,5.1,0.5)
    k_values = [10,20,30]
    num_trials = 50
    n_samples = 500
    # adata_list = []
    result_list = []
    for k in tqdm(k_values):    
        for noise in tqdm(noise_levels,leave=False):
            for dr in tqdm(density_ratios,leave=False):
                for s in tqdm(range(num_trials),leave=False):
                    df = make_circle_varying_density(noise=noise,density_ratio=dr,n_points=n_samples,seed=s)
                    results = pd.DataFrame(data=[['DM'],['LE']],columns=['Method'])
                    obs = df[['angle']].copy()
                    obs.index = [f"k{k}_noise{noise}_dr{dr}_seed{s}_cell{i}" for i in range(len(obs))]
                    adata = ad.AnnData(X=df[['x', 'y']].to_numpy(), obs=obs)
                    adata.obs['noise'] = noise
                    adata.obs['density_ratio'] = dr
                    adata.obs['seed'] = s
                    adata.obs['n_points'] = n_samples
                    adata.obs['k'] = k
                    adata.obsm['X_pca'] = adata.X

                    csr = loadAndCSR(adata, k=k)
                    lap, eigvals = laplacianEigenmaps(csr, nComponents=5)

                    kT = kTilde(csr,alpha=0.5)
                    diff, lambdas, psis = diffusion_map_from_Ktilde(kT, n_components=5)
                    lap = lap - lap.mean(axis=0)
                    adata.obsm['lap_embed'] = lap
                    diff = diff - diff.mean(axis=0)
                    adata.obsm['dm_embed'] = diff
                    dpt = np.arctan2(diff[:,0],diff[:,1])
                    dpt = align_circular_pseudotime(adata.obs['angle'],dpt)['aligned']
                    lpt = np.arctan2(lap[:,0],lap[:,1])
                    lpt = align_circular_pseudotime(adata.obs['angle'],lpt)['aligned']

                    adata.obs['lpt'] = lpt
                    adata.obs['dpt'] = dpt

                    # adata_list.append(adata)
                    results['aligned_circular_agreement'] = [aligned_circular_agreement(dpt,adata.obs['angle'].values),aligned_circular_agreement(lpt,adata.obs['angle'].values)]
                    results['mean_abs_wrapped_error'] = [mean_abs_wrapped_error(adata.obs['angle'].values,dpt),mean_abs_wrapped_error(adata.obs['angle'].values,lpt)]
                    results['radius_cv'] = [radius_cv(diff),radius_cv(lap)]
                    results['noise'] = noise
                    results['density_ratio'] = dr
                    results['seed'] = s
                    results['k'] = k
                    results['n_points'] = n_samples
                    result_list.append(results)
    # adata = ad.concat(adata_list)
    results = pd.concat(result_list)

    # adata.write_h5ad('../data/toy_datasets/density_noise_exp.h5ad')
    results.to_csv('../data/toy_datasets/density_noise_exp_metrics.csv')

if __name__ == '__main__':
    __main__()