from __future__ import annotations
import numpy as np
import pytest

import perfmath as pm

def brute_pairwise_sqeuclidean(X):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    D = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            diff = X[i] - X[j]
            D[i, j] = diff @ diff
    return D

def brute_sum_pairwise_sqeuclidean(X):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diff = X[i] - X[j]
            s += diff @ diff
    return float(s)

def brute_kmeans_inertia(X, labels, centers):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.float64)
    s = 0.0
    for i in range(X.shape[0]):
        diff = X[i] - centers[labels[i]]
        s += diff @ diff
    return float(s)

def test_pairwise_sqeuclidean_matches_brute():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 7)).astype(np.float64)
    assert np.allclose(pm.pairwise_sqeuclidean(X), brute_pairwise_sqeuclidean(X))

def test_sum_pairwise_sqeuclidean_matches_brute():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 5)).astype(np.float64)
    assert pm.sum_pairwise_sqeuclidean(X) == brute_sum_pairwise_sqeuclidean(X)

def test_rbf_kernel_properties():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 4)).astype(np.float64)
    K = pm.rbf_kernel(X, gamma=0.7)
    assert K.shape == (30, 30)
    assert np.all(K <= 1.0 + 1e-12)
    assert np.allclose(np.diag(K), 1.0)

def test_covariance_matrix_matches_numpy():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 8)).astype(np.float64)
    C1 = pm.covariance_matrix(X, ddof=1)
    C2 = np.cov(X, rowvar=False, ddof=1)
    assert np.allclose(C1, C2)

def test_kmeans_inertia_matches_brute():
    rng = np.random.default_rng(5)
    n, d, k = 100, 6, 7
    X = rng.normal(size=(n, d)).astype(np.float64)
    labels = rng.integers(0, k, size=n, dtype=np.int64)
    centers = rng.normal(size=(k, d)).astype(np.float64)
    assert pm.kmeans_inertia(X, labels, centers) == brute_kmeans_inertia(X, labels, centers)

def test_graph_energy_is_finite():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(80, 3)).astype(np.float64)
    E = pm.graph_energy(X, gamma=0.2)
    assert np.isfinite(E)