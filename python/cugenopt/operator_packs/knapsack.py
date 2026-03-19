"""
Knapsack-specific custom operators.

These operators exploit knapsack structure (value/weight ratio awareness)
to make more informed bit-flip decisions.
"""

from cugenopt.operators import CustomOperator

_KNAPSACK_GREEDY_FLIP = CustomOperator(
    name="knapsack_greedy_flip",
    code=r"""
        int row = 0;
        int sz = sol.dim2_sizes[row];
        if (sz < 2) return false;
        const float* weights = prob->d_weights;
        const float* values = prob->d_values;
        int pos = rand_int(rng, sz);
        if (sol.data[row][pos] == 0) {
            float ratio = (weights[pos] > 0.001f)
                        ? values[pos] / weights[pos] : 0.0f;
            if (ratio > 0.5f || curand_uniform(rng) < 0.3f) {
                sol.data[row][pos] = 1;
                return true;
            }
        } else {
            float ratio = (weights[pos] > 0.001f)
                        ? values[pos] / weights[pos] : 1e6f;
            if (ratio < 0.5f || curand_uniform(rng) < 0.3f) {
                sol.data[row][pos] = 0;
                return true;
            }
        }
        return false;
    """,
    encoding="binary",
    initial_weight=0.8,
)

_KNAPSACK_SWAP_RATIO = CustomOperator(
    name="knapsack_swap_ratio",
    code=r"""
        int row = 0;
        int sz = sol.dim2_sizes[row];
        if (sz < 2) return false;
        const float* weights = prob->d_weights;
        const float* values = prob->d_values;
        int in_item = -1, out_item = -1;
        for (int t = 0; t < 8; t++) {
            int p = rand_int(rng, sz);
            if (sol.data[row][p] == 1 && in_item < 0) in_item = p;
            if (sol.data[row][p] == 0 && out_item < 0) out_item = p;
            if (in_item >= 0 && out_item >= 0) break;
        }
        if (in_item < 0 || out_item < 0) return false;
        float in_ratio = (weights[in_item] > 0.001f)
                       ? values[in_item] / weights[in_item] : 1e6f;
        float out_ratio = (weights[out_item] > 0.001f)
                        ? values[out_item] / weights[out_item] : 0.0f;
        if (out_ratio > in_ratio || curand_uniform(rng) < 0.2f) {
            sol.data[row][in_item] = 0;
            sol.data[row][out_item] = 1;
            return true;
        }
        return false;
    """,
    encoding="binary",
    initial_weight=0.8,
)

knapsack_ops = [_KNAPSACK_GREEDY_FLIP, _KNAPSACK_SWAP_RATIO]
