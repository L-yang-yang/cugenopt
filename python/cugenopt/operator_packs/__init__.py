"""
Built-in operator packs for standard problems.

Each pack provides problem-aware custom operators that can significantly
improve solution quality for specific problem types. Users load them like:

    from cugenopt.operator_packs import tsp_ops
    result = cugenopt.solve_custom(..., custom_operators=tsp_ops)

Or with built-in solvers:

    result = cugenopt.solve_tsp(dist, custom_operators=tsp_ops)
"""

from cugenopt.operator_packs.tsp import tsp_ops
from cugenopt.operator_packs.knapsack import knapsack_ops
from cugenopt.operator_packs.graph_color import graph_color_ops

__all__ = ["tsp_ops", "knapsack_ops", "graph_color_ops"]
