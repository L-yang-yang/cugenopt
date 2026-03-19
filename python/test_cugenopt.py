"""Quick smoke test for cugenopt Python bindings."""

import numpy as np
import cugenopt

def test_gpu_info():
    info = cugenopt.gpu_info()
    print("GPU: %s" % info.get("name", "N/A"))
    print("  Compute: %s" % info.get("compute_capability", "N/A"))
    print("  Memory: %s" % info.get("memory", "N/A"))
    print("  Driver: %s" % info.get("driver_version", "N/A"))
    assert info["device_count"] > 0, "No GPU found"

def test_tsp():
    n = 20
    np.random.seed(0)
    coords = np.random.rand(n, 2).astype(np.float32)
    dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2))

    result = cugenopt.solve_tsp(dist, max_gen=3000, seed=42, use_aos=True)
    print(f"\nTSP-{n}: obj={result['objective']:.4f}, "
          f"time={result['elapsed_ms']:.0f}ms, gens={result['generations']}")
    route = result["solution"][0]
    assert len(route) == n
    assert set(route) == set(range(n)), "Route must be a permutation"

def test_knapsack():
    weights = np.array([2, 3, 4, 5, 1, 6, 3, 2], dtype=np.float32)
    values  = np.array([3, 4, 5, 6, 2, 8, 4, 3], dtype=np.float32)
    result = cugenopt.solve_knapsack(weights, values, capacity=12.0,
                                      max_gen=2000, seed=42)
    print(f"\nKnapsack: obj={result['objective']:.0f}, "
          f"penalty={result['penalty']:.1f}")

def test_qap():
    n = 5
    flow = np.array([
        [0, 5, 2, 4, 1],
        [5, 0, 3, 0, 2],
        [2, 3, 0, 0, 0],
        [4, 0, 0, 0, 5],
        [1, 2, 0, 5, 0],
    ], dtype=np.float32)
    dist = np.array([
        [0, 1, 2, 3, 1],
        [1, 0, 1, 2, 2],
        [2, 1, 0, 1, 3],
        [3, 2, 1, 0, 1],
        [1, 2, 3, 1, 0],
    ], dtype=np.float32)
    result = cugenopt.solve_qap(flow, dist, max_gen=3000, seed=42)
    print(f"\nQAP-{n}: obj={result['objective']:.0f}")

def test_assignment():
    n = 5
    cost = np.array([
        [9, 2, 7, 8, 6],
        [6, 4, 3, 7, 5],
        [5, 8, 1, 8, 9],
        [7, 6, 9, 4, 3],
        [2, 3, 6, 5, 7],
    ], dtype=np.float32)
    result = cugenopt.solve_assignment(cost, max_gen=2000, seed=42)
    print(f"\nAssignment-{n}: obj={result['objective']:.0f}")

def test_graph_color():
    n = 10
    adj = np.zeros((n, n), dtype=np.int32)
    edges = [(0,1),(0,4),(0,5),(1,2),(1,6),(2,3),(2,7),
             (3,4),(3,8),(4,9),(5,7),(5,8),(6,8),(6,9),(7,9)]
    for i, j in edges:
        adj[i][j] = adj[j][i] = 1
    result = cugenopt.solve_graph_color(adj, num_colors=3, max_gen=3000, seed=42)
    print(f"\nGraphColor-{n}: conflicts={result['objective']:.0f}")

if __name__ == "__main__":
    test_gpu_info()
    test_tsp()
    test_knapsack()
    test_qap()
    test_assignment()
    test_graph_color()
    print("\nAll tests passed!")
