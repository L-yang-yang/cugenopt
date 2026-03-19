"""
Built-in problem solvers — thin wrappers around compile_and_solve().

Each solve_xxx() function provides pre-written CUDA code snippets for
standard combinatorial optimization problems. Under the hood they all
call the same JIT compilation pipeline.
"""

from typing import Dict, Any, Optional
import numpy as np
from cugenopt.jit import compile_and_solve
from cugenopt.validation import (
    CuGenOptValidationError,
    validate_square_matrix,
    validate_1d,
    validate_positive_int,
)


def _solver_kwargs(kw: dict) -> dict:
    """Extract solver config kwargs from user-provided dict."""
    keys = ["pop_size", "max_gen", "time_limit", "seed", "use_aos",
            "sa_temp_init", "verbose", "cuda_arch", "framework_root",
            "custom_operators"]
    return {k: kw[k] for k in keys if k in kw}


# ============================================================
# TSP
# ============================================================

_TSP_OBJ = """
    if (idx != 0) return 0.0f;
    float total = 0.0f;
    const int* route = sol.data[0];
    int size = sol.dim2_sizes[0];
    for (int i = 0; i < size; i++)
        total += d_dist[route[i] * _n + route[(i+1) % size]];
    return total;
"""

def solve_tsp(dist_matrix: np.ndarray, **kw) -> Dict[str, Any]:
    """Solve TSP. Pass distance matrix as NxN numpy float32 array.

    Args:
        dist_matrix: NxN distance matrix (float32).
        **kw: Solver params — pop_size, max_gen, time_limit, seed, use_aos, verbose, ...

    Returns:
        Dict with objective, penalty, solution, elapsed_ms, generations, stop_reason.
    """
    dist = validate_square_matrix(dist_matrix, "dist_matrix")
    n = dist.shape[0]
    if n < 3:
        raise CuGenOptValidationError("TSP requires at least 3 cities")
    if n > 512:
        raise CuGenOptValidationError(
            f"TSP size {n} > 512 not supported yet. "
            f"Use solve_custom() for larger instances."
        )
    dim2 = 64 if n <= 64 else (256 if n <= 256 else 512)
    return compile_and_solve(
        compute_obj=_TSP_OBJ, data={"d_dist": dist},
        encoding="permutation", dim2=dim2, n=n,
        **_solver_kwargs(kw),
    )


# ============================================================
# Knapsack
# ============================================================

_KNAPSACK_OBJ = """
    if (idx != 0) return 0.0f;
    float tv = 0.0f;
    const int* sel = sol.data[0];
    int size = sol.dim2_sizes[0];
    for (int i = 0; i < size; i++)
        if (sel[i]) tv += d_values[i];
    return tv;
"""

def solve_knapsack(weights: np.ndarray, values: np.ndarray,
                   capacity: float, **kw) -> Dict[str, Any]:
    """Solve 0-1 Knapsack.

    Args:
        weights: 1D array of item weights (float32).
        values: 1D array of item values (float32).
        capacity: Knapsack capacity.
    """
    w = validate_1d(weights, "weights")
    v = validate_1d(values, "values", length=len(w))
    n = len(w)
    if capacity <= 0:
        raise CuGenOptValidationError(f"capacity must be > 0, got {capacity}")
    penalty_code = f"""
        float tw = 0.0f;
        const int* sel = sol.data[0];
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            if (sel[i]) tw += d_weights[i];
        float over = tw - {capacity}f;
        return (over > 0.0f) ? over : 0.0f;
    """
    return compile_and_solve(
        compute_obj=_KNAPSACK_OBJ, compute_penalty=penalty_code,
        data={"d_weights": w, "d_values": v},
        encoding="binary", dim2=max(32, n), n=n,
        objectives=[("maximize", 1.0)],
        **_solver_kwargs(kw),
    )


# ============================================================
# QAP
# ============================================================

_QAP_OBJ = """
    if (idx != 0) return 0.0f;
    float cost = 0.0f;
    int size = sol.dim2_sizes[0];
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            cost += d_flow[i * _n + j] * d_dist[sol.data[0][i] * _n + sol.data[0][j]];
    return cost;
"""

def solve_qap(flow_matrix: np.ndarray, dist_matrix: np.ndarray,
              **kw) -> Dict[str, Any]:
    """Solve Quadratic Assignment Problem.

    Args:
        flow_matrix: NxN flow matrix (float32).
        dist_matrix: NxN distance matrix (float32).
    """
    flow = validate_square_matrix(flow_matrix, "flow_matrix")
    dist = validate_square_matrix(dist_matrix, "dist_matrix")
    n = flow.shape[0]
    if dist.shape[0] != n:
        raise CuGenOptValidationError(
            f"flow_matrix ({n}x{n}) and dist_matrix ({dist.shape[0]}x{dist.shape[0]}) "
            f"must have the same dimensions"
        )
    return compile_and_solve(
        compute_obj=_QAP_OBJ,
        data={"d_flow": flow, "d_dist": dist},
        encoding="permutation", dim2=32, n=n,
        **_solver_kwargs(kw),
    )


# ============================================================
# Assignment
# ============================================================

_ASSIGN_OBJ = """
    if (idx != 0) return 0.0f;
    float total = 0.0f;
    const int* assign = sol.data[0];
    int size = sol.dim2_sizes[0];
    for (int i = 0; i < size; i++)
        total += d_cost[i * _n + assign[i]];
    return total;
"""

def solve_assignment(cost_matrix: np.ndarray, **kw) -> Dict[str, Any]:
    """Solve Assignment Problem.

    Args:
        cost_matrix: NxN cost matrix (float32).
    """
    cost = validate_square_matrix(cost_matrix, "cost_matrix")
    n = cost.shape[0]
    return compile_and_solve(
        compute_obj=_ASSIGN_OBJ,
        data={"d_cost": cost},
        encoding="permutation", dim2=16, n=n,
        **_solver_kwargs(kw),
    )


# ============================================================
# VRP (CVRP)
# ============================================================

def solve_vrp(dist_matrix: np.ndarray, demand: np.ndarray,
              capacity: float, num_vehicles: int, **kw) -> Dict[str, Any]:
    """Solve Capacitated VRP.

    Args:
        dist_matrix: (N+1)x(N+1) distance matrix including depot at index 0.
        demand: 1D array of customer demands (length N, excluding depot).
        capacity: Vehicle capacity.
        num_vehicles: Number of vehicles.
    """
    dist = validate_square_matrix(dist_matrix, "dist_matrix")
    n_nodes = dist.shape[0]
    n = n_nodes - 1
    dem = validate_1d(demand, "demand", length=n)
    num_vehicles = validate_positive_int(num_vehicles, "num_vehicles")
    if capacity <= 0:
        raise CuGenOptValidationError(f"capacity must be > 0, got {capacity}")
    stride = n_nodes
    max_vehicles = kw.pop("max_vehicles", num_vehicles)

    obj_code = f"""
        if (idx != 0) return 0.0f;
        float total = 0.0f;
        for (int r = 0; r < {num_vehicles}; r++) {{
            int size = sol.dim2_sizes[r];
            if (size == 0) continue;
            float dist = 0.0f;
            int prev = 0;
            for (int j = 0; j < size; j++) {{
                int node = sol.data[r][j] + 1;
                dist += d_dist[prev * {stride} + node];
                prev = node;
            }}
            dist += d_dist[prev * {stride} + 0];
            total += dist;
        }}
        return total;
    """
    penalty_code = f"""
        float penalty = 0.0f;
        int active = 0;
        for (int r = 0; r < {num_vehicles}; r++) {{
            int size = sol.dim2_sizes[r];
            if (size == 0) continue;
            active++;
            float load = 0.0f;
            for (int j = 0; j < size; j++)
                load += d_demand[sol.data[r][j]];
            if (load > {capacity}f)
                penalty += (load - {capacity}f) * 100.0f;
        }}
        if (active > {max_vehicles})
            penalty += (float)(active - {max_vehicles}) * 1000.0f;
        return penalty;
    """
    return compile_and_solve(
        compute_obj=obj_code, compute_penalty=penalty_code,
        data={"d_dist": dist, "d_demand": dem},
        encoding="permutation", dim1=num_vehicles, dim2=64, n=n,
        row_mode="partition", total_elements=n, cross_row_prob=0.3,
        **_solver_kwargs(kw),
    )


# ============================================================
# VRPTW
# ============================================================

def solve_vrptw(dist_matrix: np.ndarray, demand: np.ndarray,
                earliest: np.ndarray, latest: np.ndarray,
                service: np.ndarray, capacity: float,
                num_vehicles: int, **kw) -> Dict[str, Any]:
    """Solve VRP with Time Windows.

    Args:
        dist_matrix: (N+1)x(N+1) distance matrix including depot at index 0.
        demand: Customer demands (length N).
        earliest, latest, service: Time window arrays (length N+1, including depot).
        capacity: Vehicle capacity.
        num_vehicles: Number of vehicles.
    """
    dist = validate_square_matrix(dist_matrix, "dist_matrix")
    n_nodes = dist.shape[0]
    n = n_nodes - 1
    dem = validate_1d(demand, "demand", length=n)
    ear = validate_1d(earliest, "earliest", length=n_nodes)
    lat = validate_1d(latest, "latest", length=n_nodes)
    svc = validate_1d(service, "service", length=n_nodes)
    num_vehicles = validate_positive_int(num_vehicles, "num_vehicles")
    if capacity <= 0:
        raise CuGenOptValidationError(f"capacity must be > 0, got {capacity}")
    stride = n_nodes
    max_vehicles = kw.pop("max_vehicles", num_vehicles)

    obj_code = f"""
        if (idx != 0) return 0.0f;
        float total = 0.0f;
        for (int r = 0; r < {num_vehicles}; r++) {{
            int size = sol.dim2_sizes[r];
            if (size == 0) continue;
            float dist = 0.0f;
            int prev = 0;
            for (int j = 0; j < size; j++) {{
                int node = sol.data[r][j] + 1;
                dist += d_dist[prev * {stride} + node];
                prev = node;
            }}
            dist += d_dist[prev * {stride} + 0];
            total += dist;
        }}
        return total;
    """
    penalty_code = f"""
        float penalty = 0.0f;
        int active = 0;
        for (int r = 0; r < {num_vehicles}; r++) {{
            int size = sol.dim2_sizes[r];
            if (size == 0) continue;
            active++;
            float load = 0.0f;
            for (int j = 0; j < size; j++)
                load += d_demand[sol.data[r][j]];
            if (load > {capacity}f)
                penalty += (load - {capacity}f) * 100.0f;
            float time = 0.0f;
            int prev = 0;
            for (int j = 0; j < size; j++) {{
                int node = sol.data[r][j] + 1;
                time += d_dist[prev * {stride} + node];
                if (time < d_earliest[node]) time = d_earliest[node];
                if (time > d_latest[node])
                    penalty += (time - d_latest[node]) * 50.0f;
                time += d_service[node];
                prev = node;
            }}
            float ret = time + d_dist[prev * {stride} + 0];
            if (ret > d_latest[0])
                penalty += (ret - d_latest[0]) * 50.0f;
        }}
        if (active > {max_vehicles})
            penalty += (float)(active - {max_vehicles}) * 1000.0f;
        return penalty;
    """
    return compile_and_solve(
        compute_obj=obj_code, compute_penalty=penalty_code,
        data={"d_dist": dist, "d_demand": dem,
              "d_earliest": ear, "d_latest": lat, "d_service": svc},
        encoding="permutation", dim1=num_vehicles, dim2=64, n=n,
        row_mode="partition", total_elements=n, cross_row_prob=0.3,
        **_solver_kwargs(kw),
    )


# ============================================================
# Graph Coloring
# ============================================================

_GRAPHCOLOR_OBJ = """
    if (idx != 0) return 0.0f;
    int conflicts = 0;
    int size = sol.dim2_sizes[0];
    for (int i = 0; i < size; i++)
        for (int j = i + 1; j < size; j++)
            if (d_adj[i * _n + j] && sol.data[0][i] == sol.data[0][j])
                conflicts++;
    return (float)conflicts;
"""

def solve_graph_color(adj_matrix: np.ndarray, num_colors: int,
                      **kw) -> Dict[str, Any]:
    """Solve Graph Coloring.

    Args:
        adj_matrix: NxN adjacency matrix (int32, 1=edge, 0=no edge).
        num_colors: Number of colors available.
    """
    adj = validate_square_matrix(adj_matrix, "adj_matrix", dtype=np.int32)
    n = adj.shape[0]
    num_colors = validate_positive_int(num_colors, "num_colors")
    return compile_and_solve(
        compute_obj=_GRAPHCOLOR_OBJ,
        int_data={"d_adj": adj},
        encoding="integer", dim2=64, n=n,
        value_lower=0, value_upper=num_colors - 1,
        **_solver_kwargs(kw),
    )


# ============================================================
# Bin Packing
# ============================================================

def solve_bin_packing(item_weights: np.ndarray, max_bins: int,
                      bin_capacity: float, **kw) -> Dict[str, Any]:
    """Solve Bin Packing.

    Args:
        item_weights: 1D array of item weights (float32).
        max_bins: Maximum number of bins.
        bin_capacity: Capacity of each bin.
    """
    w = validate_1d(item_weights, "item_weights")
    n = len(w)
    max_bins = validate_positive_int(max_bins, "max_bins")
    if bin_capacity <= 0:
        raise CuGenOptValidationError(f"bin_capacity must be > 0, got {bin_capacity}")

    obj_code = f"""
        if (idx != 0) return 0.0f;
        int used = 0;
        int size = sol.dim2_sizes[0];
        for (int b = 0; b < {max_bins}; b++) {{
            bool has = false;
            for (int i = 0; i < size; i++)
                if (sol.data[0][i] == b) {{ has = true; break; }}
            if (has) used++;
        }}
        return (float)used;
    """
    penalty_code = f"""
        float penalty = 0.0f;
        int size = sol.dim2_sizes[0];
        for (int b = 0; b < {max_bins}; b++) {{
            float load = 0.0f;
            for (int i = 0; i < size; i++)
                if (sol.data[0][i] == b) load += d_weights[i];
            if (load > {bin_capacity}f)
                penalty += (load - {bin_capacity}f);
        }}
        return penalty;
    """
    return compile_and_solve(
        compute_obj=obj_code, compute_penalty=penalty_code,
        data={"d_weights": w},
        encoding="integer", dim2=64, n=n,
        value_lower=0, value_upper=max_bins - 1,
        **_solver_kwargs(kw),
    )


# ============================================================
# Load Balancing
# ============================================================

def solve_load_balance(proc_times: np.ndarray, num_machines: int,
                       **kw) -> Dict[str, Any]:
    """Solve Load Balancing (minimize makespan).

    Args:
        proc_times: 1D array of task processing times (float32).
        num_machines: Number of machines.
    """
    p = validate_1d(proc_times, "proc_times")
    n = len(p)
    num_machines = validate_positive_int(num_machines, "num_machines")

    obj_code = f"""
        if (idx != 0) return 0.0f;
        float loads[{num_machines}];
        for (int m = 0; m < {num_machines}; m++) loads[m] = 0.0f;
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            loads[sol.data[0][i]] += d_proc[i];
        float makespan = 0.0f;
        for (int m = 0; m < {num_machines}; m++)
            if (loads[m] > makespan) makespan = loads[m];
        return makespan;
    """
    return compile_and_solve(
        compute_obj=obj_code,
        data={"d_proc": p},
        encoding="integer", dim2=64, n=n,
        value_lower=0, value_upper=num_machines - 1,
        **_solver_kwargs(kw),
    )


# ============================================================
# GPU info (pure Python, no JIT needed)
# ============================================================

def gpu_info() -> Dict[str, Any]:
    """Get GPU device information via nvidia-smi."""
    import subprocess
    info = {"device_count": 0}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True
        ).strip().split("\n")[0]
        parts = [p.strip() for p in out.split(",")]
        info["device_count"] = 1
        info["name"] = parts[0]
        info["compute_capability"] = parts[1]
        info["memory"] = parts[2]
        info["driver_version"] = parts[3]
    except Exception:
        pass
    return info
