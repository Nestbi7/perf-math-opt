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

def fast_sum_pairwise_sqeuclidean_identity(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    s_norms = float(np.sum(X * X))               
    s_vec = np.sum(X, axis=0)                   
    return float(n * s_norms - float(s_vec @ s_vec))


def test_sum_pairwise_identity_matches_brute_close_but_not_bitexact():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(200, 12)).astype(np.float64)
    s_brute = brute_sum_pairwise_sqeuclidean(X)
    s_fast = fast_sum_pairwise_sqeuclidean_identity(X)
    assert np.isclose(s_fast, s_brute, rtol=1e-12, atol=1e-9)

def test_pairwise_sqeuclidean_has_zero_diag_and_nonnegative():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, 64)).astype(np.float64)
    D = pm.pairwise_sqeuclidean(X)
    assert np.all(np.diag(D) == 0.0)
    assert float(np.min(D)) >= 0.0


def test_pairwise_sqeuclidean_is_exactly_symmetric():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 32)).astype(np.float64)
    D = pm.pairwise_sqeuclidean(X)
    assert np.array_equal(D, D.T)
    
def test_sum_pairwise_sqeuclidean_adversarial_requires_dot_bitexact():
    rng = np.random.default_rng(0)
    n, d = 32, 1063
    X = (
        rng.standard_normal((n, d)) * 1e100
        + rng.standard_normal((n, d)) * 1e-100
    ).astype(np.float64)
    expected = brute_sum_pairwise_sqeuclidean(X)   # uses (diff @ diff) per pair
    got = pm.sum_pairwise_sqeuclidean(X)
    assert np.isfinite(expected)
    assert got == expected