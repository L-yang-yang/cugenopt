# cuGenOpt Python

GPU-accelerated general-purpose metaheuristic solver for combinatorial optimization.

All problems (built-in and custom) use the same JIT compilation pipeline.
First call to each problem type takes ~9s to compile; subsequent calls use cached binaries (~0.1s).

## Requirements

- NVIDIA GPU with driver installed
- `nvcc` compiler — either:
  - CUDA Toolkit installed on the system, **or**
  - `pip install nvidia-cuda-nvcc-cu12`
- Python >= 3.8

## Installation

```bash
pip install cugenopt
pip install nvidia-cuda-nvcc-cu12  # if no system CUDA Toolkit
```

## Quick Start

```python
import numpy as np
import cugenopt

# TSP: 20 cities
n = 20
coords = np.random.rand(n, 2).astype(np.float32)
dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2))

result = cugenopt.solve_tsp(dist, time_limit=5.0, seed=42)
print(f"Best distance: {result['objective']:.2f}")
print(f"Route: {result['solution'][0]}")
print(f"Time: {result['elapsed_ms']:.0f}ms, Generations: {result['generations']}")

# 0-1 Knapsack
weights = np.array([2, 3, 4, 5], dtype=np.float32)
values  = np.array([3, 4, 5, 6], dtype=np.float32)
result = cugenopt.solve_knapsack(weights, values, capacity=10.0, max_gen=2000)
print(f"Best value: {result['objective']:.0f}")

# GPU info
info = cugenopt.gpu_info()
print(f"GPU: {info['name']}, Compute: {info['compute_capability']}")
```

## Built-in Problems

| Function | Problem | Encoding |
|----------|---------|----------|
| `solve_tsp` | Traveling Salesman | Permutation |
| `solve_knapsack` | 0-1 Knapsack | Binary |
| `solve_qap` | Quadratic Assignment | Permutation |
| `solve_assignment` | Assignment | Permutation |
| `solve_vrp` | Capacitated VRP | Perm-Partition |
| `solve_vrptw` | VRP with Time Windows | Perm-Partition |
| `solve_graph_color` | Graph Coloring | Integer |
| `solve_bin_packing` | Bin Packing | Integer |
| `solve_load_balance` | Load Balancing | Integer |

## Solver Parameters

All `solve_*` functions accept keyword arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 0 (auto) | Population size (0 = auto-detect from GPU) |
| `max_gen` | 1000 | Maximum generations |
| `time_limit` | 0 (none) | Time limit in seconds |
| `seed` | 42 | Random seed |
| `use_aos` | False | Enable Adaptive Operator Selection |
| `sa_temp_init` | 0 | Simulated annealing initial temperature |
| `verbose` | False | Print progress |

## Return Value

All functions return a dict:

```python
{
    "objective": float,       # best objective value
    "penalty": float,         # constraint violation (0 = feasible)
    "solution": [np.array],   # list of row arrays
    "elapsed_ms": float,      # wall-clock time
    "generations": int,       # generations completed
    "stop_reason": str,       # "max_gen" | "time_limit" | "stagnation"
    "objectives": [float],    # all objective values
}
```

## Custom Problems (JIT)

For problems not covered by the built-in solvers, use `solve_custom()` to define
your own objective function in CUDA:

```python
import numpy as np
import cugenopt

n = 30
coords = np.random.rand(n, 2).astype(np.float32)
dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2))

result = cugenopt.solve_custom(
    compute_obj="""
        if (idx != 0) return 0.0f;
        float total = 0.0f;
        const int* route = sol.data[0];
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            total += d_dist[route[i] * _n + route[(i+1) % size]];
        return total;
    """,
    data={"d_dist": dist},
    encoding="permutation",
    dim2=64,
    n=n,
    time_limit=10.0,
)
print(f"Best: {result['objective']:.2f}")
```

The first call compiles the CUDA code (~9s). Subsequent calls with the same code
use the cached binary (~0.1s).

### solve_custom() Parameters

| Parameter | Description |
|-----------|-------------|
| `compute_obj` | CUDA code for objective function body |
| `compute_penalty` | CUDA code for penalty function body (default: `return 0.0f;`) |
| `data` | Dict of name → numpy float32 array |
| `int_data` | Dict of name → numpy int32 array |
| `encoding` | `"permutation"`, `"binary"`, or `"integer"` |
| `dim1`, `dim2` | Solution dimensions |
| `n` | Problem size |
| `objectives` | List of `(direction, weight)` tuples |
| `value_lower`, `value_upper` | Bounds for integer encoding |
| `row_mode` | `"single"`, `"fixed"`, or `"partition"` |

Use `cugenopt.clear_cache()` to remove cached compilations.
