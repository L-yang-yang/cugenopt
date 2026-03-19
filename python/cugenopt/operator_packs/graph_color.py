"""
Graph Coloring specific custom operators.

These operators exploit graph structure (adjacency awareness) to
reduce conflicts more effectively than random integer mutations.
"""

from cugenopt.operators import CustomOperator

_GC_CONFLICT_RECOLOR = CustomOperator(
    name="gc_conflict_recolor",
    code=r"""
        int row = 0;
        int sz = sol.dim2_sizes[row];
        if (sz < 2) return false;
        int n = prob->_n;
        const int* adj = prob->d_adj;
        int node = rand_int(rng, sz);
        int my_color = sol.data[row][node];
        bool has_conflict = false;
        for (int j = 0; j < sz; j++) {
            if (j != node && adj[node * n + j] && sol.data[row][j] == my_color) {
                has_conflict = true;
                break;
            }
        }
        if (!has_conflict) return false;
        int range = val_ub - val_lb + 1;
        int best_color = my_color;
        int best_conflicts = sz;
        for (int t = 0; t < 5; t++) {
            int c = val_lb + rand_int(rng, range);
            if (c == my_color) continue;
            int conflicts = 0;
            for (int j = 0; j < sz; j++)
                if (j != node && adj[node * n + j] && sol.data[row][j] == c)
                    conflicts++;
            if (conflicts < best_conflicts) {
                best_conflicts = conflicts;
                best_color = c;
            }
        }
        if (best_color == my_color) return false;
        sol.data[row][node] = best_color;
        return true;
    """,
    encoding="integer",
    initial_weight=1.5,
)

_GC_KEMPE_CHAIN = CustomOperator(
    name="gc_kempe_swap",
    code=r"""
        int row = 0;
        int sz = sol.dim2_sizes[row];
        if (sz < 3) return false;
        int n = prob->_n;
        const int* adj = prob->d_adj;
        int node = rand_int(rng, sz);
        int c1 = sol.data[row][node];
        int range = val_ub - val_lb + 1;
        int c2 = val_lb + rand_int(rng, range - 1);
        if (c2 >= c1) c2++;
        int queue[64];
        bool visited[512];
        for (int i = 0; i < sz && i < 512; i++) visited[i] = false;
        int head = 0, tail = 0;
        queue[tail++] = node;
        visited[node] = true;
        int chain_len = 0;
        while (head < tail && tail < 64 && chain_len < 16) {
            int v = queue[head++];
            int vc = sol.data[row][v];
            int swap_c = (vc == c1) ? c2 : c1;
            sol.data[row][v] = swap_c;
            chain_len++;
            for (int j = 0; j < sz; j++) {
                if (!visited[j] && j < 512 && adj[v * n + j]
                    && (sol.data[row][j] == c1 || sol.data[row][j] == c2)
                    && tail < 64) {
                    visited[j] = true;
                    queue[tail++] = j;
                }
            }
        }
        return chain_len > 0;
    """,
    encoding="integer",
    initial_weight=1.0,
)

graph_color_ops = [_GC_CONFLICT_RECOLOR, _GC_KEMPE_CHAIN]
