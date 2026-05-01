import numpy as np


def golden_spiral_sphere(n_points=10_000, upper_hemisphere=True):
    golden_angle = np.pi * (1 + np.sqrt(5))
    i = np.arange(n_points)

    theta = golden_angle * i
    z = np.linspace(1 - 1 / n_points, 1 / n_points - 1, n_points)
    r = np.sqrt(1 - z**2)

    points = np.column_stack([
        r * np.cos(theta),
        r * np.sin(theta),
        z
    ])

    if upper_hemisphere:
        points = points[points[:, 2] >= 0]

    return points


def rotation_matrix_3d(viewing_axis):
    """
    Match Schwabe/Revelio-style rotation:
    returns a 3x3 matrix that rotates the chosen viewing axis
    into the third coordinate axis.
    """
    z = np.asarray(viewing_axis, dtype=float)
    z = z / np.linalg.norm(z)

    z1, z2, z3 = z
    denom = z2**2 + z3**2

    if denom < 1e-12:
        # axis is approximately +/- x-axis
        if z1 > 0:
            return np.array([
                [0, 0, -1],
                [0, 1,  0],
                [1, 0,  0],
            ], dtype=float)
        else:
            return np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0],
            ], dtype=float)

    sin_x = z2 / denom * np.sqrt(1 - z1**2)
    cos_x = z3 / denom * np.sqrt(1 - z1**2)

    sin_y = -z1
    cos_y = np.sqrt(1 - z1**2)

    rot_x = np.array([
        [1,      0,     0],
        [0,  cos_x, sin_x],
        [0, -sin_x, cos_x],
    ])

    rot_y = np.array([
        [cos_y, 0, -sin_y],
        [0,     1,      0],
        [sin_y, 0,  cos_y],
    ])

    return (rot_y @ rot_x).T


def grid_around_axis(axis, n_points=10_000, max_radius=0.1):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    golden_angle = np.pi * (1 + np.sqrt(5))
    i = np.arange(n_points)

    radius = np.sqrt(i / n_points) * max_radius
    theta = golden_angle * i

    local_points = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros(n_points)
    ])

    R = rotation_matrix_3d(axis)
    points = local_points @ R.T + axis

    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def best_viewing_axis_minimize_phase_dispersion(
    data_3d,
    phase_labels,
    grid,
    initial_axis=None,
):
    """
    Select viewing axis whose rotated third coordinate has the least
    separation between cell-cycle phase medians.
    """
    phase_labels = np.asarray(phase_labels)
    phases = np.unique(phase_labels)

    if initial_axis is None:
        best_axis = np.array([0.0, 0.0, 1.0])
    else:
        best_axis = np.asarray(initial_axis, dtype=float)

    def score_axis(axis):
        R = rotation_matrix_3d(axis)
        rotated = data_3d @ R
        z = rotated[:, 2]

        phase_medians = np.array([
            np.median(z[phase_labels == phase])
            for phase in phases
        ])

        return np.std(phase_medians)

    best_score = score_axis(best_axis)

    for axis in grid:
        score = score_axis(axis)
        if score < best_score:
            best_score = score
            best_axis = axis

    return best_axis, best_score


def schwabe_style_optimal_rotation(
    pca_scores,
    phase_labels,
    cc_pc_mask,
    n_grid=10_000,
    refine=True,
    max_refine_loops=5,
):
    """
    Parameters
    ----------
    pca_scores : ndarray, shape (n_cells, n_pcs)
        PCA coordinates.

    phase_labels : array-like, shape (n_cells,)
        Cell-cycle phase labels, e.g. G1/S/G2/M.

    cc_pc_mask : array-like, shape (n_pcs,)
        Boolean mask indicating which PCs are cell-cycle-associated.

    Returns
    -------
    rotated_scores : ndarray, shape (n_cells, n_pcs)
        Rotated coordinates. The first two columns define the 2D trajectory.

    rotation_matrix : ndarray, shape (n_pcs, n_pcs)
        Full rotation matrix applied to PCA scores.

    trajectory_2d : ndarray, shape (n_cells, 2)
        First two rotated components.

    angle : ndarray, shape (n_cells,)
        Circular pseudotime angle in [0, 2π).

    radius : ndarray, shape (n_cells,)
        Radius in the 2D rotated trajectory.
    """
    X = np.asarray(pca_scores, dtype=float).copy()
    phase_labels = np.asarray(phase_labels)
    cc_pc_mask = np.asarray(cc_pc_mask, dtype=bool)

    n_cells, n_pcs = X.shape

    rotated = X.copy()
    full_rotation = np.eye(n_pcs)

    # Schwabe/Revelio only rotates CC-associated PCs beyond PC1/PC2
    pcs_to_rotate = np.where(cc_pc_mask)[0]
    pcs_to_rotate = pcs_to_rotate[pcs_to_rotate >= 2]

    sphere_grid = golden_spiral_sphere(n_grid, upper_hemisphere=True)

    for pc in pcs_to_rotate:
        subspace_idx = [0, 1, pc]
        data_3d = rotated[:, subspace_idx]

        first_axis, _ = best_viewing_axis_minimize_phase_dispersion(
            data_3d=data_3d,
            phase_labels=phase_labels,
            grid=sphere_grid,
            initial_axis=None,
        )

        if refine:
            # local search around best axis
            nearest_dist = np.partition(
                np.linalg.norm(sphere_grid - first_axis, axis=1),
                1
            )[1]
            radius = 3 * nearest_dist

            local_grid = grid_around_axis(
                first_axis,
                n_points=n_grid,
                max_radius=radius,
            )

            new_axis, _ = best_viewing_axis_minimize_phase_dispersion(
                data_3d=data_3d,
                phase_labels=phase_labels,
                grid=local_grid,
                initial_axis=first_axis,
            )

            n_loops = 0
            while (
                np.linalg.norm(first_axis - new_axis) > 0.75 * radius
                and n_loops < max_refine_loops
            ):
                n_loops += 1
                first_axis = new_axis

                nearest_dist = np.partition(
                    np.linalg.norm(sphere_grid - first_axis, axis=1),
                    1
                )[1]
                radius = 3 * nearest_dist

                local_grid = grid_around_axis(
                    first_axis,
                    n_points=n_grid,
                    max_radius=radius,
                )

                new_axis, _ = best_viewing_axis_minimize_phase_dispersion(
                    data_3d=data_3d,
                    phase_labels=phase_labels,
                    grid=local_grid,
                    initial_axis=first_axis,
                )

            best_axis = new_axis
        else:
            best_axis = first_axis

        R3 = rotation_matrix_3d(best_axis)

        rotated[:, subspace_idx] = rotated[:, subspace_idx] @ R3
        full_rotation[:, subspace_idx] = full_rotation[:, subspace_idx] @ R3

    trajectory_2d = rotated[:, :2]

    x = trajectory_2d[:, 0]
    y = trajectory_2d[:, 1]

    angle = np.arctan2(y, x)
    angle = np.mod(angle, 2 * np.pi)

    radius = np.sqrt(x**2 + y**2)

    return {
        "rotated_scores": rotated,
        "rotation_matrix": full_rotation,
        "trajectory_2d": trajectory_2d,
        "angle": angle,
        "radius": radius,
    }