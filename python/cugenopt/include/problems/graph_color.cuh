/**
 * graph_color.cuh - 图着色问题（Integer 编码）
 * 
 * N 个节点的图，用 k 种颜色着色。
 * 决策变量：data[0][i] ∈ [0, k-1]，表示节点 i 的颜色。
 * 目标：最小化冲突边数（相邻节点同色的边数）。
 * 
 * 验证实例：Petersen 图（10 节点 15 边，色数=3，最优冲突=0）
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

struct GraphColorProblem : ProblemBase<GraphColorProblem, 1, 64> {
    const int* d_adj;   // 邻接矩阵 [N*N]（1=相邻, 0=不相邻）
    int n;              // 节点数
    int k;              // 颜色数
    
    __device__ float calc_conflicts(const Sol& sol) const {
        int conflicts = 0;
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            for (int j = i + 1; j < size; j++)
                if (d_adj[i * n + j] && sol.data[0][i] == sol.data[0][j])
                    conflicts++;
        return (float)conflicts;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_conflicts(sol);
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        return 0.0f;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Integer;
        cfg.dim1 = 1;  cfg.dim2_default = n;
        cfg.value_lower_bound = 0;
        cfg.value_upper_bound = k - 1;
        fill_obj_config(cfg);
        return cfg;
    }
    
    size_t shared_mem_bytes() const {
        return (size_t)n * n * sizeof(int);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        int* sa = reinterpret_cast<int*>(smem);
        int total = n * n;
        for (int i = tid; i < total; i += bsz) sa[i] = d_adj[i];
        d_adj = sa;
    }
    
    static GraphColorProblem create(const int* h_adj, int n, int k) {
        GraphColorProblem prob;
        prob.n = n; prob.k = k;
        int* da;
        CUDA_CHECK(cudaMalloc(&da, sizeof(int) * n * n));
        CUDA_CHECK(cudaMemcpy(da, h_adj, sizeof(int) * n * n, cudaMemcpyHostToDevice));
        prob.d_adj = da;
        return prob;
    }
    
    void destroy() {
        if (d_adj) cudaFree(const_cast<int*>(d_adj));
        d_adj = nullptr;
    }
};
