"""
E1: GenSolver vs 通用 MIP (SCIP/CBC) — MIP 侧

目的：与 gpu.cu 对比，展示 MIP 在复杂问题上的求解时间和质量
实例：TSP (N=51,100,150), VRP (A-n32-k5)
时间预算：1s, 10s, 60s

用法：python mip.py
"""
import sys
import os
import time
from ortools.linear_solver import pywraplp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from instances import load_tsp, load_vrp, euc2d_dist_matrix, TSP_INSTANCES, VRP_INSTANCES

TIME_BUDGETS = [1, 10, 60]


def solve_tsp_mtz(dist, n, time_limit_sec, solver_id="SCIP"):
    """TSP MTZ 公式"""
    solver = pywraplp.Solver.CreateSolver(solver_id)
    if not solver:
        return float("inf"), 0.0, "error"

    x = [[solver.IntVar(0, 1, f"x_{i}_{j}") for j in range(n)] for i in range(n)]
    u = [solver.IntVar(0, n - 1, f"u_{i}") for i in range(n)]

    for i in range(n):
        solver.Add(x[i][i] == 0)
    for i in range(n):
        solver.Add(sum(x[i][j] for j in range(n)) == 1)
    for j in range(n):
        solver.Add(sum(x[i][j] for i in range(n)) == 1)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                solver.Add(u[i] - u[j] + n * x[i][j] <= n - 1)

    solver.Minimize(sum(dist[i][j] * x[i][j] for i in range(n) for j in range(n)))
    solver.SetTimeLimit(int(time_limit_sec * 1000))

    t0 = time.perf_counter()
    status = solver.Solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        reason = "optimal" if status == pywraplp.Solver.OPTIMAL else "time"
        return solver.Objective().Value(), elapsed_ms, reason
    return float("inf"), elapsed_ms, "infeasible"


def solve_vrp_mtz(dist, demands, n_nodes, n_vehicles, capacity, time_limit_sec, solver_id="SCIP"):
    """VRP MTZ 公式（容量约束 + 子回路消除）"""
    solver = pywraplp.Solver.CreateSolver(solver_id)
    if not solver:
        return float("inf"), 0.0, "error"

    n = n_nodes
    x = [[[solver.IntVar(0, 1, f"x_{k}_{i}_{j}")
            for j in range(n)] for i in range(n)] for k in range(n_vehicles)]
    u = [[solver.IntVar(0, n - 1, f"u_{k}_{i}")
          for i in range(n)] for k in range(n_vehicles)]

    # each customer visited exactly once
    for j in range(1, n):
        solver.Add(sum(x[k][i][j] for k in range(n_vehicles) for i in range(n) if i != j) == 1)

    for k in range(n_vehicles):
        # flow conservation
        for j in range(n):
            solver.Add(sum(x[k][i][j] for i in range(n) if i != j) ==
                       sum(x[k][j][i] for i in range(n) if i != j))
        # start/end at depot
        solver.Add(sum(x[k][0][j] for j in range(1, n)) <= 1)
        solver.Add(sum(x[k][j][0] for j in range(1, n)) <= 1)
        # capacity
        solver.Add(sum(demands[j] * sum(x[k][i][j] for i in range(n) if i != j)
                       for j in range(1, n)) <= capacity)
        # no self-loops
        for i in range(n):
            solver.Add(x[k][i][i] == 0)
        # MTZ subtour elimination
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    solver.Add(u[k][i] - u[k][j] + n * x[k][i][j] <= n - 1)

    solver.Minimize(sum(dist[i][j] * x[k][i][j]
                        for k in range(n_vehicles) for i in range(n) for j in range(n)))
    solver.SetTimeLimit(int(time_limit_sec * 1000))

    t0 = time.perf_counter()
    status = solver.Solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        reason = "optimal" if status == pywraplp.Solver.OPTIMAL else "time"
        return solver.Objective().Value(), elapsed_ms, reason
    return float("inf"), elapsed_ms, "infeasible"


def print_row(instance, config, obj, elapsed_ms, optimal, reason):
    if obj == float("inf"):
        print(f"{instance},{config},0,inf,0.00,{elapsed_ms:.1f},inf,0,{reason}")
    else:
        gap = (obj - optimal) / optimal * 100.0 if optimal > 0 else 0.0
        print(f"{instance},{config},0,{obj:.2f},0.00,{elapsed_ms:.1f},{gap:.2f},0,{reason}")
    sys.stdout.flush()


def main():
    print("instance,config,seed,obj,penalty,time_ms,gap_pct,generations,stop_reason")

    tsp_targets = [e for e in TSP_INSTANCES if e["optimal"] <= 6528]  # eil51, kroA100, ch150
    for entry in tsp_targets:
        inst = load_tsp(entry)
        print(f"  [e1-mip] TSP {inst['name']} (n={inst['n']})", file=sys.stderr)
        dist = euc2d_dist_matrix(inst["coords"])

        for solver_id in ["SCIP", "CBC"]:
            for t in TIME_BUDGETS:
                config = f"mip_{solver_id}_{t}s"
                obj, ms, reason = solve_tsp_mtz(dist, inst["n"], t, solver_id)
                print_row(inst["name"], config, obj, ms, inst["optimal"], reason)

    for entry in VRP_INSTANCES:
        inst = load_vrp(entry)
        print(f"  [e1-mip] VRP {inst['name']} (n={inst['n']})", file=sys.stderr)
        dist = euc2d_dist_matrix(inst["coords"])

        for solver_id in ["SCIP"]:
            for t in TIME_BUDGETS:
                config = f"mip_{solver_id}_{t}s"
                obj, ms, reason = solve_vrp_mtz(
                    dist, inst["demands"], inst["n"],
                    inst["n_vehicles"], inst["capacity"], t, solver_id)
                print_row(inst["name"], config, obj, ms, inst["optimal"], reason)


if __name__ == "__main__":
    main()
