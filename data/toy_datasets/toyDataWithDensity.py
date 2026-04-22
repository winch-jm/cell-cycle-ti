# making toy datasets with varying density along two axes for demo implementations

import numpy as np
import pandas as pd

#### Concentric circle dataset with varying density regions

def make_circle_varying_density(n_points=500, noise=0.05, dense_region=(0, np.pi), density_ratio=5, seed=42):
    """
    Inputs:
    n_points: total number of points
    noise: standard deviation of Gaussian noise added to x, y
    dense_region: (start_angle, end_angle) in radians where points are denser
    density_ratio: how many times denser the dense region is vs the sparse region
    seed: random seed

    Output:
    dataframe: with columns x, y, angle, target
    """
    rng = np.random.default_rng(seed)

    arc_fraction = (dense_region[1] - dense_region[0]) / (2 * np.pi)
    n_dense = int(n_points * arc_fraction * density_ratio / (arc_fraction * density_ratio + (1 - arc_fraction)))
    n_sparse = n_points - n_dense

    dense_angles = rng.uniform(dense_region[0], dense_region[1], n_dense)
    sparse_angles = rng.uniform(dense_region[1], dense_region[0] + 2 * np.pi, n_sparse) % (2 * np.pi)
    angles = np.concatenate([dense_angles, sparse_angles])

    x = np.cos(angles) + rng.normal(0, noise, n_points)
    y = np.sin(angles) + rng.normal(0, noise, n_points)

    df = pd.DataFrame({"x": x, "y": y, "angle": angles, "target": 0})
    return df


#### Ellipse dataset with varying density regions

def make_ellipse_varying_density(n_points=500, noise=0.05, a=2.0, b=1.0, dense_region=(0, np.pi), density_ratio=5, seed=42):
    """
    Inputs:
    n_points: total number of points
    noise: standard deviation of Gaussian noise added to x, y
    a: semi-major axis (x radius)
    b: semi-minor axis (y radius)
    dense_region: (start_angle, end_angle) in radians where points are denser
    density_ratio: how many times denser the dense region is vs the sparse region
    seed: random seed

    Output:
    dataFrame with columns x, y, angle, target
    """
    rng = np.random.default_rng(seed)

    arc_fraction = (dense_region[1] - dense_region[0]) / (2 * np.pi)
    n_dense = int(n_points * arc_fraction * density_ratio /
                  (arc_fraction * density_ratio + (1 - arc_fraction)))
    n_sparse = n_points - n_dense

    dense_angles = rng.uniform(dense_region[0], dense_region[1], n_dense)
    sparse_angles = rng.uniform(dense_region[1], dense_region[0] + 2 * np.pi, n_sparse) % (2 * np.pi)
    angles = np.concatenate([dense_angles, sparse_angles])

    x = a * np.cos(angles) + rng.normal(0, noise, n_points)
    y = b * np.sin(angles) + rng.normal(0, noise, n_points)

    df = pd.DataFrame({"x": x, "y": y, "angle": angles, "target": 0})

    return df

