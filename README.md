# Performance Optimization of Numerical Kernels (Academic Case Study)

## Overview

This repository contains an academic case study focused on the performance optimization of numerical Python code.  
The project was developed as part of a university-level assignment on software performance engineering and reproducibility for optimization cases.

The goal of the code is to improve the execution speed and efficiency of computationally intensive mathematical routines without changing their functional behavior, which we use in several different scenarios.

---

## Problem Context

In many university and research settings, numerical code is initially written in a straightforward, loop-based style to prioritize correctness and clarity.  We use this kind of code mostly in problems of transport, and recently we tried to useit in Biology.
However, despite being simple routines that all our colleagues who are not from the exact sciences can understand, such implementations often become performance bottlenecks when applied to larger datasets.

This project tries to show basic uses in a realistic scenario, as we use them, where:
- A research group uses Python and NumPy for numerical experiments
- Initial implementations rely heavily on Python loops
- Execution time becomes prohibitive for moderate-to-large input sizes
- The task is to optimize the code while preserving exact outputs

The assignment emphasizes:
- Algorithmic complexity reduction
- Vectorization using NumPy
- Reproducibility via Docker
- Automated and machine-readable testing

---

## Technical Focus

The optimized components include:
- Pairwise squared Euclidean distance computations
- Kernel-based similarity calculations
- Statistical estimators (covariance/correlation)
- Objective functions used in clustering and graph-based methods

Key optimization techniques:
- Replacing nested Python loops with vectorized NumPy operations
- Using algebraic identities to reduce time complexity
- Leveraging BLAS-backed matrix operations
- Avoiding unnecessary intermediate allocations

---

## Testing and Evaluation

All correctness checks are implemented as unit tests executed via `pytest`.

To support automated evaluation, test execution is wrapped in a bash script that produces machine-readable JSON output describing individual test results.
