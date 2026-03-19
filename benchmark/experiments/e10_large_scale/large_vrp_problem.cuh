#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

// 支持大规模 VRP（最多 256 个客户，16 辆车）
struct LargeVRPProblem : ProblemBase<LargeVRPProblem, 16, 256> {
    const float* d_dist;
    const float* d_demand;
    const float* h_dist;
    const float* h_demand;
    int n;
    float capacity;
    int num_vehicles;
    int max_vehicles;
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f}
    };
    
    __device__ float compute_obj(int obj_idx, const Sol& s) const {
        float total = 0;
        for (int v = 0; v < num_vehicles; v++) {
            int route_len = s.dim2_sizes[v];
            if (route_len == 0) continue;
            
            // 从depot到第一个客户（客户编号需要+1，因为0是depot）
            int first_node = s.data[v][0] + 1;
            total += d_dist[0 * (n+1) + first_node];
            
            // 路径内部
            int prev = first_node;
            for (int i = 1; i < route_len; i++) {
                int node = s.data[v][i] + 1;
                total += d_dist[prev * (n+1) + node];
                prev = node;
            }
            
            // 最后一个客户回depot
            total += d_dist[prev * (n+1) + 0];
        }
        return total;
    }
    
    __device__ float compute_penalty(const Sol& s) const {
        float penalty = 0;
        for (int v = 0; v < num_vehicles; v++) {
            float load = 0;
            for (int i = 0; i < s.dim2_sizes[v]; i++) {
                load += d_demand[s.data[v][i]];
            }
            if (load > capacity) {
                penalty += (load - capacity) * 100.0f;
            }
        }
        return penalty;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = num_vehicles;
        cfg.dim2_default = 0;  // Partition 模式下由框架自动分配
        fill_obj_config(cfg);
        cfg.cross_row_prob = 0.3f;
        cfg.row_mode = RowMode::Partition;
        cfg.total_elements = n;  // 总共有 n 个客户需要分配到各车辆
        return cfg;
    }
    
    // 可选：覆盖 working_set_bytes 用于 L2 cache 感知
    size_t working_set_bytes() const {
        return (size_t)(n + 1) * (n + 1) * sizeof(float) + (size_t)n * sizeof(float);
    }
    
    static LargeVRPProblem create(const float* h_dist_matrix, const float* h_demand_array,
                                   int num_customers, float vehicle_capacity,
                                   int num_veh, int max_veh) {
        LargeVRPProblem prob;
        prob.n = num_customers;
        prob.capacity = vehicle_capacity;
        prob.num_vehicles = num_veh;
        prob.max_vehicles = max_veh;
        prob.h_dist = h_dist_matrix;
        prob.h_demand = h_demand_array;
        
        size_t dist_size = (size_t)(num_customers + 1) * (num_customers + 1) * sizeof(float);
        size_t demand_size = (size_t)num_customers * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&prob.d_dist, dist_size));
        CUDA_CHECK(cudaMalloc(&prob.d_demand, demand_size));
        CUDA_CHECK(cudaMemcpy((void*)prob.d_dist, h_dist_matrix, dist_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)prob.d_demand, h_demand_array, demand_size, cudaMemcpyHostToDevice));
        
        return prob;
    }
    
    void destroy() {
        if (d_dist) cudaFree((void*)d_dist);
        if (d_demand) cudaFree((void*)d_demand);
        d_dist = nullptr;
        d_demand = nullptr;
    }
    
    // Multi-GPU support
    LargeVRPProblem* clone_to_device(int target_gpu) const {
        int orig_device;
        CUDA_CHECK(cudaGetDevice(&orig_device));
        CUDA_CHECK(cudaSetDevice(target_gpu));
        
        // 分配设备内存并拷贝数据到目标 GPU
        float* dd;
        float* ddem;
        size_t dist_size = (size_t)(n + 1) * (n + 1) * sizeof(float);
        size_t demand_size = (size_t)n * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&dd, dist_size));
        CUDA_CHECK(cudaMalloc(&ddem, demand_size));
        CUDA_CHECK(cudaMemcpy(dd, h_dist, dist_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ddem, h_demand, demand_size, cudaMemcpyHostToDevice));
        
        // 恢复原设备
        CUDA_CHECK(cudaSetDevice(orig_device));
        
        // 创建新的 Problem 实例（在 host 端）
        LargeVRPProblem* new_prob = new LargeVRPProblem();
        new_prob->n = n;
        new_prob->capacity = capacity;
        new_prob->num_vehicles = num_vehicles;
        new_prob->max_vehicles = max_vehicles;
        new_prob->h_dist = h_dist;
        new_prob->h_demand = h_demand;
        new_prob->d_dist = dd;
        new_prob->d_demand = ddem;
        
        return new_prob;
    }
};
