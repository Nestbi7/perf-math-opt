from __future__ import annotations
import numpy as np

def covariance_matrix(X: np.ndarray, ddof: int = 1) -> np.ndarray:
    """
    Sample covariance matrix (d, d).
    Baseline intentionally uses slow loops.
    Optimized with vectorized matrix multiplication.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    C = Xc.T @ Xc
    return C / float(n - ddof)

def corrcoef_matrix(X: np.ndarray) -> np.ndarray:
    """
    Correlation matrix (d, d).
    """
    C = covariance_matrix(X, ddof=1)
    std = np.sqrt(np.diag(C))
    return C / (std[:, None] * std[None, :])
