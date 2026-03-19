/**
 * tsp_xlarge.cuh - 超大规模 TSP 问题定义 (最多 512 城市)
 * 
 * 继承 ProblemBase，逻辑与 tsp_large.cuh 一致，D2=512
 * 注意：距离矩阵 512×512×4B = 1MB，远超 48KB shared memory
 *       因此 shared_mem_bytes() 返回 0，距离矩阵留在 global memory
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

struct TSPXLargeProblem : ProblemBase<TSPXLargeProblem, 1, 512> {
    const float* d_dist;
    const float* h_dist;  // host 端距离矩阵（用于 init_relation_matrix）
    int n;
    
    __device__ float calc_total_distance(const Sol& sol) const {
        float total = 0.0f;
        const int* route = sol.data[0];
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            total += d_dist[route[i] * n + route[(i + 1) % size]];
        return total;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_total_distance(sol);
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const { return 0.0f; }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = 1;  cfg.dim2_default = n;
        fill_obj_config(cfg);
        return cfg;
    }
    
    // 距离矩阵太大，不放 shared memory
    size_t shared_mem_bytes() const { return 0; }
    __device__ void load_shared(char*, int, int) {}
    
    size_t working_set_bytes() const {
        return (size_t)n * n * sizeof(float);
    }
    
    // 用距离矩阵初始化 G/O 先验：距离近 → 分数高
    void init_relation_matrix(float* G, float* O, int N) const {
        if (!h_dist || N != n) return;
        // 找最大距离用于归一化
        float max_d = 0.0f;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (h_dist[i * N + j] > max_d) max_d = h_dist[i * N + j];
        if (max_d <= 0.0f) return;
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                // 距离近 → G 高（分组倾向强）
                float proximity = 1.0f - h_dist[i * N + j] / max_d;
                G[i * N + j] = proximity * 0.3f;  // 初始信号不要太强，留空间给 EMA
                // 距离近 → O 也给一点信号（对称的，不偏向任何方向）
                O[i * N + j] = proximity * 0.1f;
            }
        }
    }
    
    int heuristic_matrices(HeuristicMatrix* out, int max_count) const {
        if (max_count < 1 || !h_dist) return 0;
        out[0] = {h_dist, n};
        return 1;
    }
    
    static TSPXLargeProblem create(const float* h_dist_ptr, int n) {
        TSPXLargeProblem prob;
        prob.n = n;
        prob.h_dist = h_dist_ptr;  // 保留 host 指针
        float* dd;
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * n * n));
        CUDA_CHECK(cudaMemcpy(dd, h_dist_ptr, sizeof(float) * n * n, cudaMemcpyHostToDevice));
        prob.d_dist = dd;
        return prob;
    }
    
    void destroy() {
        if (d_dist) { cudaFree(const_cast<float*>(d_dist)); d_dist = nullptr; }
        h_dist = nullptr;
    }
};
