#pragma once
#include "types.cuh"
#include <vector>
#include <algorithm>
#include <numeric>

namespace heuristic_init {

// 单行排列：所有行填相同排列
template<typename Sol>
static void build_sorted_permutation(Sol& sol, const std::vector<int>& order,
                                     int dim1, int dim2) {
    for (int r = 0; r < dim1; r++) {
        sol.dim2_sizes[r] = dim2;
        for (int c = 0; c < dim2; c++)
            sol.data[r][c] = order[c];
    }
    sol.penalty = 0.0f;
    for (int i = 0; i < MAX_OBJ; i++) sol.objectives[i] = 0.0f;
}

// Partition 模式：排列均匀切分到 dim1 行，元素不重复
template<typename Sol>
static void build_partition_from_order(Sol& sol, const std::vector<int>& order,
                                       int dim1, int total_elements) {
    int idx = 0;
    for (int r = 0; r < dim1; r++) {
        int count = total_elements / dim1;
        if (r < total_elements % dim1) count++;
        sol.dim2_sizes[r] = count;
        for (int c = 0; c < count; c++)
            sol.data[r][c] = order[idx++];
    }
    sol.penalty = 0.0f;
    for (int i = 0; i < MAX_OBJ; i++) sol.objectives[i] = 0.0f;
}

template<typename Sol>
std::vector<Sol> build_from_matrices(const HeuristicMatrix* matrices, int num_matrices,
                                     int dim1, int dim2, EncodingType encoding,
                                     bool partition_mode = false, int total_elements = 0) {
    std::vector<Sol> results;
    if (encoding != EncodingType::Permutation) return results;
    int elem_count = partition_mode ? total_elements : dim2;
    if (num_matrices <= 0 || elem_count <= 0) return results;

    auto make_sol = [&](const std::vector<int>& order) {
        Sol sol{};
        if (partition_mode)
            build_partition_from_order(sol, order, dim1, total_elements);
        else
            build_sorted_permutation(sol, order, dim1, dim2);
        return sol;
    };

    for (int m = 0; m < num_matrices; m++) {
        const float* mat = matrices[m].data;
        int N = matrices[m].N;
        if (!mat || N < elem_count) continue;

        std::vector<float> row_sum(N, 0.0f);
        std::vector<float> col_sum(N, 0.0f);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                row_sum[i] += mat[i * N + j];
                col_sum[j] += mat[i * N + j];
            }

        // 对于 Partition (VRPTW)，距离矩阵含 depot (index 0)，
        // 排序只针对客户 (index 1..N-1)，输出值为 0-based 客户编号
        std::vector<int> idx;
        if (partition_mode && N > elem_count) {
            for (int i = 1; i <= elem_count; i++) idx.push_back(i);
        } else {
            idx.resize(elem_count);
            std::iota(idx.begin(), idx.end(), 0);
        }

        auto to_customer = [&](std::vector<int>& order) {
            if (partition_mode && N > elem_count) {
                for (auto& v : order) v -= 1;
            }
        };

        // row_sum ascending
        {
            auto order = idx;
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return row_sum[a] < row_sum[b]; });
            to_customer(order);
            results.push_back(make_sol(order));
        }
        // row_sum descending
        {
            auto order = idx;
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return row_sum[a] > row_sum[b]; });
            to_customer(order);
            results.push_back(make_sol(order));
        }
        // col_sum ascending
        {
            auto order = idx;
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return col_sum[a] < col_sum[b]; });
            to_customer(order);
            results.push_back(make_sol(order));
        }
        // col_sum descending
        {
            auto order = idx;
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return col_sum[a] > col_sum[b]; });
            to_customer(order);
            results.push_back(make_sol(order));
        }
    }
    return results;
}

} // namespace heuristic_init
