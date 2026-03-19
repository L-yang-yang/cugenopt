/**
 * knapsack.cuh - 0-1 背包问题
 * 
 * 继承 ProblemBase，使用 ObjDef 目标注册机制
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

struct KnapsackProblem : ProblemBase<KnapsackProblem, 1, 32> {
    // 问题数据（d_weights 是物品重量，非目标权重）
    const float* d_weights;
    const float* d_values;
    float capacity;
    int n;
    
    // ---- 目标计算 ----
    __device__ float calc_total_value(const Sol& sol) const {
        float tv = 0.0f;
        const int* sel = sol.data[0];
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            if (sel[i]) tv += d_values[i];
        return tv;
    }
    
    // ---- 目标定义（OBJ_DEFS 与 compute_obj 必须一一对应）----
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Maximize, 1.0f, 0.0f},   // case 0: calc_total_value
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_total_value(sol);   // OBJ_DEFS[0]
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        float tw = 0.0f;
        const int* sel = sol.data[0];
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            if (sel[i]) tw += d_weights[i];
        float over = tw - capacity;
        return (over > 0.0f) ? over : 0.0f;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Binary;
        cfg.dim1 = 1;  cfg.dim2_default = n;
        fill_obj_config(cfg);
        return cfg;
    }
    
    // ---- shared memory 接口 ----
    size_t shared_mem_bytes() const {
        return 2 * (size_t)n * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sw = reinterpret_cast<float*>(smem);
        float* sv = sw + n;
        for (int i = tid; i < n; i += bsz) { sw[i] = d_weights[i]; sv[i] = d_values[i]; }
        d_weights = sw;
        d_values = sv;
    }
    
    static KnapsackProblem create(const float* hw, const float* hv, int n, float cap) {
        KnapsackProblem prob;
        prob.n = n; prob.capacity = cap;
        float *dw, *dv;
        CUDA_CHECK(cudaMalloc(&dw, sizeof(float)*n));
        CUDA_CHECK(cudaMalloc(&dv, sizeof(float)*n));
        CUDA_CHECK(cudaMemcpy(dw, hw, sizeof(float)*n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv, hv, sizeof(float)*n, cudaMemcpyHostToDevice));
        prob.d_weights = dw; prob.d_values = dv;
        return prob;
    }
    
    void destroy() {
        if (d_weights) cudaFree(const_cast<float*>(d_weights));
        if (d_values)  cudaFree(const_cast<float*>(d_values));
        d_weights = nullptr; d_values = nullptr;
    }
};
