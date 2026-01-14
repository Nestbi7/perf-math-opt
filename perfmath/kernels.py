from __future__ import annotations
import numpy as np

def pairwise_sqeuclidean(X: np.ndarray) -> np.ndarray:
    """
    Return the full (n, n) matrix of squared Euclidean distances.
    X: (n, d) float64.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    D = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        xi = X[i]
        for j in range(n):
            diff = xi - X[j]
            D[i, j] = float(diff @ diff)
    return D

def sum_pairwise_sqeuclidean(X: np.ndarray) -> float:
    """
    Return sum_{i<j} ||xi-xj||^2 as float64.
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