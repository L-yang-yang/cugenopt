"""
TSP-specific custom operators.

These operators exploit TSP structure (distance matrix, route continuity)
to perform more effective local search than generic permutation operators.
"""

from cugenopt.operators import CustomOperator

_TSP_2OPT_DELTA = CustomOperator(
    name="tsp_2opt_delta",
    code=r"""
        int row = 0;
        int sz = sol.dim2_sizes[row];
        if (sz < 4) return false;
        int n = prob->_n;
        const float* dist = prob->d_dist;
        int i = rand_int(rng, sz);
        int j = rand_int(rng, sz - 2);
        if (j >= i) j += 2; else if (j == i - 1) j = (i + sz - 1) % sz;
        if (i > j) { int t = i; i = j; j = t; }
        if (j - i < 2 || j - i >= sz - 1) return false;
        int a = sol.data[row][i], b = sol.data[row][(i+1) % sz];
        int c = sol.data[row][j], d = sol.data[row][(j+1) % sz];
        float old_cost = dist[a * n + b] + dist[c * n + d];
        float new_cost = dist[a * n + c] + dist[b * n + d];
        if (new_cost >= old_cost) return false;
        ops::perm_reverse(sol.data[row], i + 1, j);
        return true;
    """,
    encoding="permutation",
    initial_weight=1.5,
)

_TSP_OR_OPT_DELTA = CustomOperator(
    name="tsp_or_opt_delta",
    code=r"""
        int row = 0;
        int sz = sol.dim2_sizes[row];
        if (sz < 5) return false;
        int n = prob->_n;
        const float* dist = prob->d_dist;
        int seg_len = 1 + rand_int(rng, 3);
        if (seg_len >= sz - 1) return false;
        int from_pos = rand_int(rng, sz - seg_len);
        int prev = sol.data[row][(from_pos + sz - 1) % sz];
        int seg_start = sol.data[row][from_pos];
        int seg_end = sol.data[row][from_pos + seg_len - 1];
        int next = sol.data[row][(from_pos + seg_len) % sz];
        float remove_cost = dist[prev * n + seg_start]
                          + dist[seg_end * n + next]
                          - dist[prev * n + next];
        int best_to = -1;
        float best_delta = 0.0f;
        for (int t = 0; t < 8; t++) {
            int to = rand_int(rng, sz - seg_len);
            if (to >= from_pos && to < from_pos + seg_len) continue;
            int tp = sol.data[row][to];
            int tn = sol.data[row][(to + 1) % sz];
            float insert_cost = dist[tp * n + seg_start]
                              + dist[seg_end * n + tn]
                              - dist[tp * n + tn];
            float delta = insert_cost - remove_cost;
            if (delta < best_delta) { best_delta = delta; best_to = to; }
        }
        if (best_to < 0) return false;
        ops::perm_or_opt(sol.data[row], sz, from_pos, best_to, seg_len);
        return true;
    """,
    encoding="permutation",
    initial_weight=1.0,
)

_TSP_NODE_INSERT_DELTA = CustomOperator(
    name="tsp_node_insert_delta",
    code=r"""
        int row = 0;
        int sz = sol.dim2_sizes[row];
        if (sz < 4) return false;
        int n = prob->_n;
        const float* dist = prob->d_dist;
        int from_pos = rand_int(rng, sz);
        int node = sol.data[row][from_pos];
        int prev = sol.data[row][(from_pos + sz - 1) % sz];
        int next = sol.data[row][(from_pos + 1) % sz];
        float remove_save = dist[prev * n + node] + dist[node * n + next]
                          - dist[prev * n + next];
        int best_to = -1;
        float best_delta = 0.0f;
        for (int t = 0; t < 8; t++) {
            int to = rand_int(rng, sz - 1);
            if (to >= from_pos) to++;
            int tp = sol.data[row][to];
            int tn = sol.data[row][(to + 1) % sz];
            float insert_cost = dist[tp * n + node] + dist[node * n + tn]
                              - dist[tp * n + tn];
            float delta = insert_cost - remove_save;
            if (delta < best_delta) { best_delta = delta; best_to = to; }
        }
        if (best_to < 0) return false;
        ops::perm_insert(sol.data[row], from_pos, best_to, sz);
        return true;
    """,
    encoding="permutation",
    initial_weight=1.0,
)

tsp_ops = [_TSP_2OPT_DELTA, _TSP_OR_OPT_DELTA, _TSP_NODE_INSERT_DELTA]
