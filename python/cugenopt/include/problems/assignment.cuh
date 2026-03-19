/**
 * assignment.cuh - 指派问题
 * 
 * 继承 ProblemBase，使用 ObjDef 目标注册机制
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

struct AssignmentProblem : ProblemBase<AssignmentProblem, 1, 16> {
    const float* d_cost;
    const float* h_cost;  // host 端成本矩阵（用于 init_relation_matrix）
    int n;
    
    // ---- 目标计算 ----
    __device__ float calc_total_cost(const Sol& sol) const {
        float total = 0.0f;
        const int* assign = sol.data[0];
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            total += d_cost[i * n + assign[i]];
        return total;
    }
    
    // ---- 目标定义（OBJ_DEFS 与 compute_obj 必须一一对应）----
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},   // case 0: calc_total_cost
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_total_cost(sol);   // OBJ_DEFS[0]
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
    
    // ---- shared memory 接口 ----
    static constexpr size_t SMEM_LIMIT = 48 * 1024;
    
    size_t shared_mem_bytes() const {
        size_t need = (size_t)n * n * sizeof(float);
        return need <= SMEM_LIMIT ? need : 0;
    }
    
    size_t working_set_bytes() const {
        return (size_t)n * n * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sc = reinterpret_cast<float*>(smem);
        int total = n * n;
        for (int i = tid; i < total; i += bsz) sc[i] = d_cost[i];
        d_cost = sc;
    }
    
    // 成本先验：task j 和 task k 如果被相似 agent 偏好，G 值高
    // O 矩阵：task j 在位置 i 成本低 → O[j][k] 略高（j 倾向排在 k 前面的位置）
    void init_relation_matrix(float* G, float* O, int N) const {
        if (!h_cost || N != n) return;
        // 对每个 task，构建成本向量，task 间余弦相似度 → G
        // 简化：成本列向量的相关性
        float max_c = 0.0f;
        for (int i = 0; i < N * N; i++)
            if (h_cost[i] > max_c) max_c = h_cost[i];
        if (max_c <= 0.0f) return;
        
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++) {
                if (j == k) continue;
                // G: 两个 task 的成本向量越相似 → 越可能互换
                float dot = 0.0f, nj = 0.0f, nk = 0.0f;
                for (int i = 0; i < N; i++) {
                    float cj = h_cost[i * N + j] / max_c;
                    float ck = h_cost[i * N + k] / max_c;
                    dot += cj * ck;
                    nj += cj * cj;
                    nk += ck * ck;
                }
                float denom = sqrtf(nj) * sqrtf(nk);
                float sim = (denom > 1e-6f) ? dot / denom : 0.0f;
                G[j * N + k] = sim * 0.2f;
                O[j * N + k] = sim * 0.05f;
            }
    }
    
    static AssignmentProblem create(const float* hc, int n) {
        AssignmentProblem prob;
        prob.n = n;
        prob.h_cost = hc;
        float* dc;
        CUDA_CHECK(cudaMalloc(&dc, sizeof(float)*n*n));
        CUDA_CHECK(cudaMemcpy(dc, hc, sizeof(float)*n*n, cudaMemcpyHostToDevice));
        prob.d_cost = dc;
        return prob;
    }
    
    void destroy() {
        if (d_cost) { cudaFree(const_cast<float*>(d_cost)); d_cost = nullptr; }
        h_cost = nullptr;
    }
};
