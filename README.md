# cuGenOpt

> **A GPU-Accelerated General-Purpose Metaheuristic Framework for Combinatorial Optimization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**Paper**: [cuGenOpt: A GPU-Accelerated General-Purpose Metaheuristic Framework for Combinatorial Optimization](https://arxiv.org/abs/XXXX.XXXXX) *(Coming soon)*

---

## Overview

cuGenOpt is a high-performance, problem-agnostic GPU metaheuristic framework designed for combinatorial optimization. It provides:

- **Generic Solution Encodings**: Permutation, Binary, Integer, and Partition representations
- **Adaptive Operator Selection (AOS)**: Runtime weight adjustment via exponential moving average
- **Three-Layer Adaptive Architecture**: Static priors (L1) + Runtime AOS (L3) for cold-start avoidance
- **GPU Memory Hierarchy Optimization**: L2 cache-aware population sizing and adaptive shared memory management
- **Multi-GPU Support**: Independent parallel solving with automatic device management
- **Python API + CUDA C++**: High-level interface with JIT compilation for custom problems

### Key Features

| Feature | Description |
|---------|-------------|
| **12+ Problem Types** | TSP, VRP, VRPTW, Knapsack, QAP, JSP, Assignment, Graph Coloring, Bin Packing, and more |
| **Adaptive Search** | EMA-driven operator weight adjustment during runtime |
| **Problem Profiling** | Automatic initial strategy selection based on problem characteristics |
| **Memory-Aware** | Automatic population sizing based on GPU L2 cache capacity |
| **Multi-Objective** | Weighted sum and lexicographic optimization modes |
| **Cross-Platform** | Unified workflow on Linux and Windows |

---

## Quick Start

### Option 1: Python API (Recommended)

```bash
pip install cugenopt
pip install nvidia-cuda-nvcc-cu12  # If system CUDA Toolkit not available
```

**Solve Built-in Problems:**

```python
import numpy as np
import cugenopt

# Solve TSP
dist = np.random.rand(50, 50).astype(np.float32)
dist = (dist + dist.T) / 2  # Make symmetric
result = cugenopt.solve_tsp(dist, time_limit=10.0)
print(f"Best tour length: {result['best_obj']}")
print(f"Tour: {result['best_solution']}")
```

**Define Custom Problems with JIT:**

```python
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
    dim2=50,
    n=50,
    time_limit=10.0
)
```

### Option 2: CUDA C++ Direct Usage

```bash
cd prototype
make tsp
./tsp
```

Define your own problem by inheriting `ProblemBase` and implementing `compute_obj` / `compute_penalty`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Python API Layer                     │
│  (Built-in Problems + JIT Compiler for Custom Problems) │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                 Core Framework (CUDA C++)               │
│  • Adaptive Solver (L1 Priors + L3 Runtime AOS)        │
│  • Operator Registry (Swap, Reverse, Insert, LNS, ...)  │
│  • Population Management (Elite + Diversity)            │
│  • Multi-GPU Coordinator                                │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│              GPU Execution Engine                       │
│  • L2 Cache-Aware Memory Management                     │
│  • Adaptive Shared Memory Allocation                    │
│  • CUDA Kernels (Population-level + Neighborhood-level) │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
generic_solver/
├── prototype/              # Core framework (header-only .cuh files)
│   ├── core/              #   Solver, operators, population, types
│   └── problems/          #   12+ problem implementations
├── python/                 # Python wrapper (pip install cugenopt)
│   ├── cugenopt/          #   Python package (built-ins + JIT compiler)
│   └── tests/             #   Test suite
├── benchmark/              # Experiments and benchmarks
│   ├── experiments/       #   E0-E13: 14 experiment groups
│   ├── data/              #   Standard instances (TSPLIB, Solomon, QAPLIB)
│   └── results/           #   Experimental reports
├── paper_v3_en/            # Paper source (LaTeX)
├── STATUS.md               # Project status and roadmap
└── README.md               # This file
```

---

## Performance Highlights

### Benchmark Results

| Problem | Instance | cuGenOpt | Best Known | Gap |
|---------|----------|----------|------------|-----|
| TSP | kroA100 | 21,282 | 21,282 | 0.00% |
| TSP | kroA200 | 29,368 | 29,368 | 0.00% |
| QAP | nug12 | 578 | 578 | **0.00%** (Optimal) |
| VRPTW | C101 | 828.94 | 828.94 | 0.00% |
| VRPTW | R101 | 1,650.80 | 1,645.79 | 0.30% |

### GPU Scalability

| GPU | Memory Bandwidth | TSP n=1000 Speedup |
|-----|------------------|-------------------|
| T4 | 300 GB/s | 1.0× (baseline) |
| V100 | 900 GB/s | 1.6× |
| A800 | 1,935 GB/s | 3.6× |

*Memory-bound workload: performance scales linearly with bandwidth.*

### Multi-GPU Effectiveness

| Problem | Single GPU | 2× GPU | 4× GPU | Improvement |
|---------|-----------|--------|--------|-------------|
| TSP n=1000 | 7,542,668 | 7,277,989 | 7,236,344 | **3.51%** |
| QAP n=100 | 1,520,516 | 1,502,084 | 1,498,404 | **1.45%** |

*With CUDA Graph enabled. Larger problems benefit more from parallel exploration.*

---

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
- Recommended: 8GB+ GPU memory for large-scale problems

### Software
- CUDA Toolkit 11.0+
- Python 3.8+ (for Python API)
- GCC 7.5+ or MSVC 2019+ (for C++ compilation)

---

## Installation

### Python Package

```bash
pip install cugenopt
```

### Build from Source

```bash
git clone https://github.com/L-yang-yang/cugenopt.git
cd cugenopt/python
pip install -e .
```

### CUDA C++ Only

```bash
cd prototype
make all
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [STATUS.md](STATUS.md) | Project status, roadmap, and design decisions |
| [Python API Guide](python/README.md) | Detailed Python API documentation |
| [Benchmark Design](benchmark/DESIGN.md) | Experimental methodology |
| [Paper](paper_v3_en/) | Full technical details and evaluation |

---

## Citation

If you use cuGenOpt in your research, please cite:

```bibtex
@article{liu2026cugenopt,
  title={cuGenOpt: A GPU-Accelerated General-Purpose Metaheuristic Framework for Combinatorial Optimization},
  author={Liu, Yuyang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Contact

**Yuyang Liu**  
Independent Researcher, Shenzhen, China  
Email: 15251858055@163.com

---

## Acknowledgments

This work was conducted as independent research. Special thanks to the open-source community for providing excellent tools and libraries that made this project possible.
