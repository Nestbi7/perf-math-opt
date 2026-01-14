from __future__ import annotations
import time
import numpy as np

from perfmath import (
    pairwise_sqeuclidean,
    sum_pairwise_sqeuclidean,
    rbf_kernel,
    covariance_matrix,
    kmeans_inertia,
    graph_energy,
)

def timed(name, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    print(f"{name:24s}  {t1-t0:9.4f} sec")
    return out

def main():
    rng = np.random.default_rng(0)

    # Workload sizes: keep feasible on laptops but clearly show bottlenecks.
    n = 2500
    d = 32
    X = rng.normal(size=(n, d)).astype(np.float64)

    gamma = 0.15

    print("=== Benchmark workload ===")
    print(f"n={n}, d={d}, gamma={gamma}")

    # 1) Pairwise distances
    D = timed("pairwise_sqeuclidean", pairwise_sqeuclidean, X)
    s1 = timed("sum_pairwise_sqeuclid", sum_pairwise_sqeuclidean, X)

    # 2) Kernel
    K = timed("rbf_kernel", rbf_kernel, X, gamma)

    # 3) Stats
    C = timed("covariance_matrix", covariance_matrix, X, 1)

    # 4) Objective functions
    k = 20
    labels = rng.integers(0, k, size=n, dtype=np.int64)
    centers = rng.normal(size=(k, d)).astype(np.float64)
    inertia = timed("kmeans_inertia", kmeans_inertia, X, labels, centers)
    E = timed("graph_energy", graph_energy, X, gamma)

    # cheap sanity checks (not full tests)
    assert D.shape == (n, n)
    assert K.shape == (n, n)
    assert C.shape == (d, d)
    assert np.isfinite(s1) and np.isfinite(inertia) and np.isfinite(E)
    print("Sanity checks: OK")

if __name__ == "__main__":
    main()