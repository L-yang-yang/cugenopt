#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

// 支持大规模 TSP（最多 512 个城市）
struct LargeTSPProblem : ProblemBase<LargeTSPProblem, 1, 512> {
    const float* d_dist;
    const float* h_dist;
    int n;
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f}
    };
    
    __device__ float compute_obj(int obj_idx, const Sol& s) const {
        float total = 0;
        for (int i = 0; i < n - 1; i++) {
            int from = s.data[0][i];
            int to = s.data[0][i + 1];
            total += d_dist[from * n + to];
        }
        total += d_dist[s.data[0][n - 1] * n + s.data[0][0]];
        return total;
    }
    
    __device__ float compute_penalty(const Sol& s) const {
        return 0.0f;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = 1;
        cfg.dim2_default = n;
        fill_obj_config(cfg);
        return cfg;
    }
    
    // 可选：覆盖 working_set_bytes 用于 L2 cache 感知
    size_t working_set_bytes() const {
        return (size_t)n * n * sizeof(float);
    }
    
    static LargeTSPProblem create(const float* h_dist_matrix, int num_cities) {
        LargeTSPProblem prob;
        prob.n = num_cities;
        prob.h_dist = h_dist_matrix;
        
        size_t dist_size = (size_t)num_cities * num_cities * sizeof(float);
        CUDA_CHECK(cudaMalloc(&prob.d_dist, dist_size));
        CUDA_CHECK(cudaMemcpy((void*)prob.d_dist, h_dist_matrix, dist_size, cudaMemcpyHostToDevice));
        
        return prob;
    }
    
    void destroy() {
        if (d_dist) {
            cudaFree((void*)d_dist);
            d_dist = nullptr;
        }
    }
    
    // Multi-GPU support
    LargeTSPProblem* clone_to_device(int target_gpu) const {
        int orig_device;
        CUDA_CHECK(cudaGetDevice(&orig_device));
        CUDA_CHECK(cudaSetDevice(target_gpu));
        
        // 分配设备内存并拷贝距离矩阵到目标 GPU
        float* dd;
        size_t dist_size = (size_t)n * n * sizeof(float);
        CUDA_CHECK(cudaMalloc(&dd, dist_size));
        CUDA_CHECK(cudaMemcpy(dd, h_dist, dist_size, cudaMemcpyHostToDevice));
        
        // 恢复原设备
        CUDA_CHECK(cudaSetDevice(orig_device));
        
        // 创建新的 Problem 实例（在 host 端）
        LargeTSPProblem* new_prob = new LargeTSPProblem();
        new_prob->n = n;
        new_prob->h_dist = h_dist;
        new_prob->d_dist = dd;
        
        return new_prob;
    }
};
