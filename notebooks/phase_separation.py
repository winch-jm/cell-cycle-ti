
import numpy as np
import pandas as pd

def phase_circular_concentration(theta, phases):
    """
    Higher is better.
    Measures how tightly each phase concentrates around an angular position.
    """
    theta = np.asarray(theta)
    phases = np.asarray(phases)

    rows = []
    for phase in np.unique(phases):
        vals = theta[phases == phase]

        R = np.abs(np.mean(np.exp(1j * vals)))  # mean resultant length
        circ_var = 1 - R

        rows.append({
            "phase": phase,
            "n": len(vals),
            "mean_resultant_length": R,
            "circular_variance": circ_var,
        })

    df = pd.DataFrame(rows)
    weighted_R = np.average(df["mean_resultant_length"], weights=df["n"])

    return weighted_R, df

def angular_phase_margin_score(
    theta,
    phases,
    phase_order,
    min_cells_per_phase=5,
    return_details=False,
):
    """
    Lower is better.

    Measures whether phase centroids are separated in the expected cyclic order.
    Penalizes overlapping or reversed phase centroids.
    """
    theta = np.asarray(theta)
    phases = np.asarray(phases)

    centroids = []
    ns = []
    within = []

    for phase in phase_order:
        vals = theta[phases == phase]
        if len(vals) < min_cells_per_phase:
            raise ValueError(f"Too few cells for phase {phase}: {len(vals)}")

        z = np.mean(np.exp(1j * vals))
        mu = np.angle(z)
        R = np.abs(z)

        centroids.append(mu)
        ns.append(len(vals))
        within.append(1 - R)

    centroids = np.array(centroids)
    ns = np.array(ns)
    within = np.array(within)

    # unwrap centroids in the expected phase order
    unwrapped = np.unwrap(centroids)

    # allow either clockwise or counterclockwise orientation
    diffs_forward = np.diff(np.r_[unwrapped, unwrapped[0] + 2 * np.pi])
    diffs_reverse = np.diff(np.r_[unwrapped[::-1], unwrapped[::-1][0] + 2 * np.pi])

    # good separation means all adjacent centroid gaps are positive and not tiny
    min_gap_forward = np.min(diffs_forward)
    min_gap_reverse = np.min(diffs_reverse)

    best_min_gap = max(min_gap_forward, min_gap_reverse)

    weighted_within = np.average(within, weights=ns)

    # Lower is better: high within spread is bad, small/negative gap is bad
    score = weighted_within - best_min_gap

    if return_details:
        return score, {
            "weighted_within_dispersion": weighted_within,
            "best_min_adjacent_gap": best_min_gap,
            "centroids": dict(zip(phase_order, centroids)),
            "within_dispersion_by_phase": dict(zip(phase_order, within)),
        }

    return score