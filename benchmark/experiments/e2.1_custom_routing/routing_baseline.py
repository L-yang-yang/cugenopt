"""
E2.1: 自定义路径规划 — OR-Tools Routing baseline

OR-Tools Routing 的两个建模限制：
  A. 无法表达路径内优先级偏序约束（Dimension 只支持累积约束）
  B. 无法使用负载依赖的非线性边成本（ArcCostEvaluator 只接受 from/to）

因此只能求解标准 CVRP，然后事后：
  - 统计优先级违规数量
  - 用非线性公式重新计算真实成本

用法：python routing_baseline.py
"""
import sys
import os
import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from instances import load_vrp, euc2d_dist_matrix, VRP_INSTANCES

TIME_BUDGETS = [1, 10, 60]

# 与 gpu.cu 一致的优先级分配
# 客户 0-9: high(2), 10-20: medium(1), 21-30: low(0)
PRIORITIES = (
    [2] * 10 +   # customers 0-9: high
    [1] * 11 +   # customers 10-20: medium
    [0] * 10     # customers 21-30: low
)


def count_priority_violations(routes, priorities):
    """统计所有路径中的优先级违规数量。
    违规定义：同一路径内，高优先级客户出现在低优先级客户之后。
    """
    violations = 0
    for route in routes:
        min_prio_seen = 3
        for node in route:
            p = priorities[node]
            if p > min_prio_seen:
                violations += 1
            if p < min_prio_seen:
                min_prio_seen = p
    return violations


def calc_nonlinear_cost(routes, dist, demands, capacity):
    """用非线性公式重新计算路径成本。
    cost(edge) = dist(i,j) * (1.0 + 0.3 * (load/capacity)²)
    与 gpu.cu 中 NonlinearCostVRPProblem::compute_route_nonlinear_cost 一致。
    dist 矩阵含 depot（index 0），客户编号 0-based → node = cust + 1。
    """
    total = 0.0
    for route in routes:
        load = 0.0
        prev = 0  # depot
        for cust in route:
            node = cust + 1
            load += demands[node]
            ratio = load / capacity
            total += dist[prev][node] * (1.0 + 0.3 * ratio * ratio)
            prev = node
        total += dist[prev][0]  # 返回 depot，空载系数 1.0
    return total


def solve_cvrp_routing(dist, demands, n, n_vehicles, capacity, time_limit_sec):
    """标准 CVRP 求解（无优先级约束）"""
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_callback(from_idx, to_idx):
        return dist[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_id = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_id)

    def demand_callback(idx):
        return demands[manager.IndexToNode(idx)]

    demand_id = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_id, 0, [capacity] * n_vehicles, True, "Cap")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = time_limit_sec

    t0 = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not solution:
        return float("inf"), elapsed_ms, [], "infeasible"

    obj = solution.ObjectiveValue()
    routes = []
    for v in range(n_vehicles):
        route = []
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route.append(node - 1)  # 转为 0-based 客户编号
            idx = solution.Value(routing.NextVar(idx))
        routes.append(route)

    return obj, elapsed_ms, routes, "time"


def print_row(instance, config, obj, elapsed_ms, optimal, violations, reason):
    if obj == float("inf"):
        print(f"{instance},{config},0,inf,0.00,{elapsed_ms:.1f},inf,0,{reason}")
    else:
        gap = (obj - optimal) / optimal * 100.0 if optimal > 0 else 0.0
        print(f"{instance},{config},0,{obj:.2f},0.00,{elapsed_ms:.1f},"
              f"{gap:.2f},0,{reason}_v{violations}")
    sys.stdout.flush()


def main():
    print("instance,config,seed,obj,penalty,time_ms,gap_pct,generations,stop_reason")

    for entry in VRP_INSTANCES:
        inst = load_vrp(entry)
        n_customers = inst["n"] - 1
        print(f"  [e2.1-routing] VRP {inst['name']} (n={inst['n']})",
              file=sys.stderr)
        dist = euc2d_dist_matrix(inst["coords"])
        demands_full = [0] + list(inst["demands"])  # index 0 = depot
        priorities = PRIORITIES[:n_customers]

        for t in TIME_BUDGETS:
            obj, ms, routes, reason = solve_cvrp_routing(
                dist, demands_full,
                inst["n"], inst["n_vehicles"], inst["capacity"], t)

            violations = count_priority_violations(routes, priorities) if routes else -1

            # 场景 A: 优先级约束
            print_row(
                f"{inst['name']}-prio",
                f"routing_GLS_{t}s",
                obj, ms, inst["optimal"], violations, reason)

            # 标准 VRP baseline
            print_row(
                f"{inst['name']}-std",
                f"routing_GLS_{t}s",
                obj, ms, inst["optimal"], 0, reason)

            # 场景 B: 非线性成本（用 OR-Tools 的解重新计算真实成本）
            if routes:
                nl_cost = calc_nonlinear_cost(
                    routes, dist, demands_full, inst["capacity"])
                print_row(
                    f"{inst['name']}-nlcost",
                    f"routing_GLS_{t}s",
                    nl_cost, ms, 0, 0, reason)
            else:
                print_row(
                    f"{inst['name']}-nlcost",
                    f"routing_GLS_{t}s",
                    float("inf"), ms, 0, 0, reason)


if __name__ == "__main__":
    main()
