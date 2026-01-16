from __future__ import annotations
import numpy as np
from .kernels import pairwise_sqeuclidean

def kmeans_inertia(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    Sum of squared distances to assigned center.
    Baseline uses Python loop.
    IMPORTANT: Keep exact accumulation order for bit-exact tests.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.float64)
    total = 0.0
    for i in range(X.shape[0]):
        c = centers[labels[i]]
        diff = X[i] - c
        total += float(diff @ diff)
    return float(total)

def graph_energy(X: np.ndarray, gamma: float) -> float:
    """
    A simple "energy" based on a fully-connected RBF similarity graph:
      E = sum_{i<j} exp(-gamma * ||xi-xj||^2)
    """
    D = pairwise_sqeuclidean(X)
    K = np.exp(-gamma * D, dtype=np.float64)
    return float(np.triu(K, k=1).sum())
