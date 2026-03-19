"""
E2: GenSolver vs 专用求解器 (OR-Tools Routing) — Routing 侧

目的：与 gpu.cu 对比，展示专用求解器的质量优势
实例：TSP (全部 TSPLIB), VRP (A-n32-k5)
时间预算：1s, 5s, 10s, 30s, 60s

用法：python routing.py [tsp|vrp|all]
"""
import sys
import os
import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from instances import load_tsp, load_vrp, euc2d_dist_matrix, TSP_INSTANCES, VRP_INSTANCES

TSP_TIME_BUDGETS = [1, 5, 10, 30, 60]
VRP_TIME_BUDGETS = [1, 5, 10, 30]


def solve_tsp_routing(dist, n, time_limit_sec):
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_callback(from_idx, to_idx):
        return dist[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_id = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_id)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_sec

    t0 = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    obj = solution.ObjectiveValue() if solution else float("inf")
    return obj, elapsed_ms


def solve_cvrp_routing(dist, demands, n, n_vehicles, capacity, time_limit_sec):
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_callback(from_idx, to_idx):
        return dist[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_id = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_id)

    def demand_callback(idx):
        return demands[manager.IndexToNode(idx)]

    demand_id = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_id, 0, [capacity] * n_vehicles, True, "Cap")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_sec

    t0 = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    obj = solution.ObjectiveValue() if solution else float("inf")
    return obj, elapsed_ms


def print_row(instance, config, obj, elapsed_ms, optimal):
    if obj == float("inf"):
        print(f"{instance},{config},0,inf,0.00,{elapsed_ms:.1f},inf,0,time")
    else:
        gap = (obj - optimal) / optimal * 100.0 if optimal > 0 else 0.0
        print(f"{instance},{config},0,{obj:.2f},0.00,{elapsed_ms:.1f},{gap:.2f},0,time")
    sys.stdout.flush()


def run_tsp():
    for entry in TSP_INSTANCES:
        inst = load_tsp(entry)
        print(f"  [e2-routing] TSP {inst['name']} (n={inst['n']})", file=sys.stderr)
        dist = euc2d_dist_matrix(inst["coords"])
        for t in TSP_TIME_BUDGETS:
            obj, ms = solve_tsp_routing(dist, inst["n"], t)
            print_row(inst["name"], f"routing_GLS_{t}s", obj, ms, inst["optimal"])


def run_vrp():
    for entry in VRP_INSTANCES:
        inst = load_vrp(entry)
        print(f"  [e2-routing] VRP {inst['name']} (n={inst['n']})", file=sys.stderr)
        dist = euc2d_dist_matrix(inst["coords"])
        for t in VRP_TIME_BUDGETS:
            obj, ms = solve_cvrp_routing(
                dist, inst["demands"], inst["n"],
                inst["n_vehicles"], inst["capacity"], t)
            print_row(inst["name"], f"routing_GLS_{t}s", obj, ms, inst["optimal"])


def main():
    print("instance,config,seed,obj,penalty,time_ms,gap_pct,generations,stop_reason")
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in ("all", "tsp"):
        run_tsp()
    if target in ("all", "vrp"):
        run_vrp()


if __name__ == "__main__":
    main()
