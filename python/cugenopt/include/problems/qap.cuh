/**
 * qap.cuh - 二次分配问题 (Quadratic Assignment Problem)
 * 
 * N 个设施分配到 N 个位置（排列编码）。
 * 决策变量：data[0][i] = 设施 i 分配到的位置。
 * 目标：Minimize sum(flow[i][j] * dist[perm[i]][perm[j]])
 * 
 * 验证实例：自定义 5x5
 *   flow: 设施间的物流量
 *   dist: 位置间的距离
 *   已知最优 = 58
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

struct QAPProblem : ProblemBase<QAPProblem, 1, 32> {
    const float* d_flow;    // 物流量矩阵 [N*N]
    const float* d_dist;    // 距离矩阵 [N*N]
    int n;
    
    __device__ float calc_cost(const Sol& sol) const {
        float cost = 0.0f;
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                cost += d_flow[i * n + j] * d_dist[sol.data[0][i] * n + sol.data[0][j]];
        return cost;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_cost(sol);
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        return 0.0f;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = 1;  cfg.dim2_default = n;
        fill_obj_config(cfg);
        return cfg;
    }
    
    size_t shared_mem_bytes() const {
        return 2 * (size_t)n * n * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sf = reinterpret_cast<float*>(smem);
        float* sd = sf + n * n;
        int total = n * n;
        for (int i = tid; i < total; i += bsz) { sf[i] = d_flow[i]; sd[i] = d_dist[i]; }
        d_flow = sf;
        d_dist = sd;
    }
    
    static QAPProblem create(const float* h_flow, const float* h_dist, int n) {
        QAPProblem prob;
        prob.n = n;
        float *df, *dd;
        CUDA_CHECK(cudaMalloc(&df, sizeof(float) * n * n));
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * n * n));
        CUDA_CHECK(cudaMemcpy(df, h_flow, sizeof(float) * n * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dd, h_dist, sizeof(float) * n * n, cudaMemcpyHostToDevice));
        prob.d_flow = df; prob.d_dist = dd;
        return prob;
    }
    
    void destroy() {
        if (d_flow) cudaFree(const_cast<float*>(d_flow));
        if (d_dist) cudaFree(const_cast<float*>(d_dist));
        d_flow = nullptr; d_dist = nullptr;
    }
};
