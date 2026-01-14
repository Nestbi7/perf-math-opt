from __future__ import annotations
import numpy as np

def covariance_matrix(X: np.ndarray, ddof: int = 1) -> np.ndarray:
    """
    Sample covariance matrix (d, d).
    Baseline intentionally uses slow loops.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mu = X.mean(axis=0)
    C = np.zeros((d, d), dtype=np.float64)
    for i in range(n):
        v = X[i] - mu
        for a in range(d):
            for b in range(d):
                C[a, b] += v[a] * v[b]
    return C / float(n - ddof)

def corrcoef_matrix(X: np.ndarray) -> np.ndarray:
    """
    Correlation matrix (d, d).
    """
    C = covariance_matrix(X, ddof=1)
    std = np.sqrt(np.diag(C))
    return C / (std[:, None] * std[None, :])