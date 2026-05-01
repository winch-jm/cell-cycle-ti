import numpy as np

TWOPI = 2 * np.pi

def wrap_angle(x):
    return np.mod(x, TWOPI)

def circ_mean(alpha):
    return np.arctan2(np.sin(alpha).sum(), np.cos(alpha).sum())

def mean_cosine_agreement(theta1, theta2):
    # both assumed already wrapped
    return np.mean(np.cos(theta1 - theta2))

def align_circular_pseudotime(theta_full, theta_sub, allow_reflection=True):
    """
    Align theta_sub to theta_full by optimal rotation, optionally allowing reversal.

    Parameters
    ----------
    theta_full : array-like, shape (n,)
        Reference angles in radians.
    theta_sub : array-like, shape (n,)
        Angles to align in radians.
    allow_reflection : bool
        If True, also test reversed direction.

    Returns
    -------
    result : dict
        {
            "aligned": aligned angles,
            "delta": optimal rotation,
            "sign": +1 or -1,
            "score": mean cosine agreement
        }
    """
    theta_full = wrap_angle(np.asarray(theta_full))
    theta_sub = wrap_angle(np.asarray(theta_sub))

    signs = [1, -1] if allow_reflection else [1]
    best = None

    for s in signs:
        theta_trial = wrap_angle(s * theta_sub)

        # optimal rotation = circular mean of angle differences
        diffs = wrap_angle(theta_full - theta_trial)
        delta = circ_mean(diffs)

        aligned = wrap_angle(theta_trial + delta)
        score = mean_cosine_agreement(theta_full, aligned)

        candidate = {
            "aligned": aligned,
            "delta": delta,
            "sign": s,
            "score": score,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best