import numpy as np

def emd1d(p, q):
    """Simple 1D EMD (Wasserstein-1) between histograms p and q (same length).
    Assumes p, q are nonnegative and sum > 0; returns float.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-9)
    q = q / (q.sum() + 1e-9)
    cdf_diff = np.cumsum(p - q)
    return float(np.mean(np.abs(cdf_diff)))