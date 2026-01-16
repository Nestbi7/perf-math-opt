from __future__ import annotations
import numpy as np

def pairwise_sqeuclidean(X: np.ndarray) -> np.ndarray:
    """
    Return the full (n, n) matrix of squared Euclidean distances.
    X: (n, d) float64.
    """
    X = np.asarray(X, dtype=np.float64)

    # Vectorized identity:
    # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 * <xi, xj>
    sq_norms = np.einsum("ij,ij->i", X, X).reshape(-1, 1)
    G = X @ X.T
    D = sq_norms + sq_norms.T - 2.0 * G

    np.fill_diagonal(D, 0.0)
    np.maximum(D, 0.0, out=D)
    return D

def sum_pairwise_sqeuclidean(X: np.ndarray) -> float:
    """
    Return sum_{i<j} ||xi-xj||^2 as float64.
    IMPORTANT: Keep exact accumulation order for bit-exact tests.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    total = 0.0
    for i in range(n):
        xi = X[i]
        for j in range(i + 1, n):
            diff = xi - X[j]
            total += float(diff @ diff)
    return float(total)

def rbf_kernel(X: np.ndarray, gamma: float) -> np.ndarray:
    """
    K_ij = exp(-gamma * ||xi-xj||^2)
    """
    D = pairwise_sqeuclidean(X)
    return np.exp(-gamma * D, dtype=np.float64)
