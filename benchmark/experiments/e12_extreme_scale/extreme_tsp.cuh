#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

// 极大规模 TSP（最多 2048 个城市）
struct ExtremeTSPProblem : ProblemBase<ExtremeTSPProblem, 1, 2048> {
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
    
    size_t working_set_bytes() const {
        return (size_t)n * n * sizeof(float);
    }
    
    static ExtremeTSPProblem create(const float* h_dist_matrix, int num_cities) {
        ExtremeTSPProblem prob;
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
    
    ExtremeTSPProblem* clone_to_device(int target_gpu) const {
        int orig_device;
        CUDA_CHECK(cudaGetDevice(&orig_device));
        CUDA_CHECK(cudaSetDevice(target_gpu));
        
        float* dd;
        size_t dist_size = (size_t)n * n * sizeof(float);
        CUDA_CHECK(cudaMalloc(&dd, dist_size));
        CUDA_CHECK(cudaMemcpy(dd, h_dist, dist_size, cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaSetDevice(orig_device));
        
        ExtremeTSPProblem* new_prob = new ExtremeTSPProblem();
        new_prob->n = n;
        new_prob->h_dist = h_dist;
        new_prob->d_dist = dd;
        
        return new_prob;
    }
};
