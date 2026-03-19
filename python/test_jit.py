"""Test JIT custom problem compilation and solving."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from cugenopt.jit import compile_and_solve, clear_cache


def test_custom_tsp():
    """Custom TSP via JIT — same problem as layer 1, verifies JIT pipeline."""
    n = 20
    np.random.seed(0)
    coords = np.random.rand(n, 2).astype(np.float32)
    dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2)).astype(np.float32)

    result = compile_and_solve(
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
        max_gen=3000,
        seed=42,
        use_aos=True,
        verbose=True,
    )
    print(f"\n[JIT] TSP-{n}: obj={result['objective']:.4f}, "
          f"time={result['elapsed_ms']:.0f}ms, gens={result['generations']}")
    route = result["solution"][0]
    assert len(route) == n, f"Route length {len(route)} != {n}"
    assert set(route) == set(range(n)), "Route must be a permutation"


def test_custom_knapsack():
    """Custom 0-1 Knapsack via JIT."""
    n = 8
    weights = np.array([2, 3, 4, 5, 1, 6, 3, 2], dtype=np.float32)
    values = np.array([3, 4, 5, 6, 2, 8, 4, 3], dtype=np.float32)
    capacity = 12.0

    result = compile_and_solve(
        compute_obj="""
            if (idx != 0) return 0.0f;
            float tv = 0.0f;
            const int* sel = sol.data[0];
            int size = sol.dim2_sizes[0];
            for (int i = 0; i < size; i++)
                if (sel[i]) tv += d_values[i];
            return tv;
        """,
        compute_penalty=f"""
            float tw = 0.0f;
            const int* sel = sol.data[0];
            int size = sol.dim2_sizes[0];
            for (int i = 0; i < size; i++)
                if (sel[i]) tw += d_weights[i];
            float over = tw - {capacity}f;
            return (over > 0.0f) ? over : 0.0f;
        """,
        data={"d_weights": weights, "d_values": values},
        encoding="binary",
        dim2=32,
        n=n,
        objectives=[("maximize", 1.0)],
        max_gen=2000,
        seed=42,
    )
    print(f"\n[JIT] Knapsack: obj={result['objective']:.0f}, "
          f"penalty={result['penalty']:.1f}")


def test_custom_graph_color():
    """Custom Graph Coloring via JIT with int data."""
    n = 10
    k = 3
    adj = np.zeros((n, n), dtype=np.int32)
    edges = [(0,1),(0,4),(0,5),(1,2),(1,6),(2,3),(2,7),
             (3,4),(3,8),(4,9),(5,7),(5,8),(6,8),(6,9),(7,9)]
    for i, j in edges:
        adj[i][j] = adj[j][i] = 1

    result = compile_and_solve(
        compute_obj="""
            if (idx != 0) return 0.0f;
            int conflicts = 0;
            int size = sol.dim2_sizes[0];
            for (int i = 0; i < size; i++)
                for (int j = i + 1; j < size; j++)
                    if (d_adj[i * _n + j] && sol.data[0][i] == sol.data[0][j])
                        conflicts++;
            return (float)conflicts;
        """,
        int_data={"d_adj": adj},
        encoding="integer",
        dim2=64,
        n=n,
        value_lower=0,
        value_upper=k - 1,
        max_gen=3000,
        seed=42,
    )
    print(f"\n[JIT] GraphColor-{n}: conflicts={result['objective']:.0f}")


def test_cache_hit():
    """Second call with same code should use cached binary."""
    n = 10
    np.random.seed(1)
    dist = np.random.rand(n, n).astype(np.float32)
    dist = (dist + dist.T) / 2

    import time
    t0 = time.time()
    r1 = compile_and_solve(
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
        encoding="permutation", dim2=64, n=n, max_gen=500, seed=42,
    )
    t1 = time.time()

    r2 = compile_and_solve(
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
        encoding="permutation", dim2=64, n=n, max_gen=500, seed=42,
    )
    t2 = time.time()

    first_time = t1 - t0
    second_time = t2 - t1
    print(f"\n[JIT] Cache test: 1st={first_time:.1f}s, 2nd={second_time:.1f}s "
          f"(speedup: {first_time/max(second_time, 0.001):.1f}x)")
    assert second_time < first_time, "Cached run should be faster"


if __name__ == "__main__":
    clear_cache()
    test_custom_tsp()
    test_custom_knapsack()
    test_custom_graph_color()
    test_cache_hit()
    print("\nAll JIT tests passed!")
