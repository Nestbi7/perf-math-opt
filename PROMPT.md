## Performance Optimization Task

Please work on improving the overall performance and execution speed of this repository. The primary goal is to reduce runtime and/or memory footprint for large numerical workloads while preserving **exact output correctness**.

You may focus your optimization efforts on one or more performance-sensitive areas, including but not limited to:
- Pairwise distance and similarity computations
- Batch statistics (covariance/correlation) for large datasets
- Objective computations used in optimization loops (e.g., clustering inertia / graph energy)

The following conditions apply:
1. Improving the performance of at least one performance-sensitive area is enough, as performance evaluations will be conducted collectively across the benchmark suite.
2. Optimizations may be implemented either directly in high-level functions or indirectly by accelerating lower-level computational kernels on which they depend.
3. Optimization efforts should prioritize achieving the largest feasible efficiency gains.
4. Output must remain exactly the same for all supported inputs and all unit tests must pass.

Reproduction:
- Run `python repro.py` to measure baseline performance and validate outputs.
- Run `pytest -q` for correctness.

Notes:
- The repository uses NumPy for numerical computing.
- Do not change the meaning of any public function.