"""
cuGenOpt — GPU-accelerated general-purpose metaheuristic solver

All problems (built-in and custom) use the same JIT compilation pipeline.
First call to each problem type takes ~8s to compile; subsequent calls are cached.

Usage:
    import numpy as np
    import cugenopt

    dist = np.random.rand(20, 20).astype(np.float32)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)

    result = cugenopt.solve_tsp(dist, time_limit=5.0, seed=42)
    print(f"Best distance: {result['objective']:.2f}")
    print(f"Route: {result['solution'][0]}")
"""

from cugenopt.builtins import (
    solve_tsp,
    solve_knapsack,
    solve_qap,
    solve_assignment,
    solve_vrp,
    solve_vrptw,
    solve_graph_color,
    solve_bin_packing,
    solve_load_balance,
    gpu_info,
)
from cugenopt.jit import compile_and_solve as solve_custom, clear_cache
from cugenopt.validation import CuGenOptValidationError, CuGenOptCompileError
from cugenopt.operators import CustomOperator

__version__ = "0.2.0"

__all__ = [
    "solve_tsp",
    "solve_knapsack",
    "solve_qap",
    "solve_assignment",
    "solve_vrp",
    "solve_vrptw",
    "solve_graph_color",
    "solve_bin_packing",
    "solve_load_balance",
    "gpu_info",
    "solve_custom",
    "clear_cache",
    "CuGenOptValidationError",
    "CuGenOptCompileError",
    "CustomOperator",
]
