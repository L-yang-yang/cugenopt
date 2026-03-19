#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

/**
 * 三目标 VRP: 最小化总距离 + 最小化车辆数 + 最小化最大路径长度（负载均衡）
 * 
 * 目标1: 总距离（主要目标）
 * 目标2: 使用的车辆数（次要目标）
 * 目标3: 最大路径长度（负载均衡目标）
 * 
 * 测试场景：
 * - Weighted 模式：权重配置 [0.6, 0.2, 0.2]
 * - Lexicographic 模式：优先级 [距离, 车辆, 最大路径]
 */
struct TriObjectiveVRP : ProblemBase<TriObjectiveVRP, 16, 64> {
    const float* d_dist;
    const float* d_demand;
    int n;
    float capacity;
    int max_vehicles;
    
    // 三目标定义
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},  // 目标0: 最小化总距离
        {ObjDir::Minimize, 1.0f, 0.0f},  // 目标1: 最小化车辆数
        {ObjDir::Minimize, 1.0f, 0.0f},  // 目标2: 最小化最大路径长度
    };
    
    static constexpr int NUM_OBJ = 3;
    
    __device__ float compute_obj(int obj_idx, const Sol& s) const {
        if (obj_idx == 0) {
            // 目标1: 总距离
            float total = 0.0f;
            for (int v = 0; v < max_vehicles; v++) {
                int route_len = s.dim2_sizes[v];
                if (route_len == 0) continue;
                
                int first_node = s.data[v][0] + 1;
                total += d_dist[0 * (n+1) + first_node];
                
                int prev = first_node;
                for (int i = 1; i < route_len; i++) {
                    int node = s.data[v][i] + 1;
                    total += d_dist[prev * (n+1) + node];
                    prev = node;
                }
                
                total += d_dist[prev * (n+1) + 0];
            }
            return total;
        } else if (obj_idx == 1) {
            // 目标2: 使用的车辆数
            int used = 0;
            for (int v = 0; v < max_vehicles; v++) {
                if (s.dim2_sizes[v] > 0) used++;
            }
            return (float)used;
        } else {
            // 目标3: 最大路径长度（负载均衡）
            float max_route_dist = 0.0f;
            for (int v = 0; v < max_vehicles; v++) {
                int route_len = s.dim2_sizes[v];
                if (route_len == 0) continue;
                
                float route_dist = 0.0f;
                int first_node = s.data[v][0] + 1;
                route_dist += d_dist[0 * (n+1) + first_node];
                
                int prev = first_node;
                for (int i = 1; i < route_len; i++) {
                    int node = s.data[v][i] + 1;
                    route_dist += d_dist[prev * (n+1) + node];
                    prev = node;
                }
                
                route_dist += d_dist[prev * (n+1) + 0];
                
                if (route_dist > max_route_dist) {
                    max_route_dist = route_dist;
                }
            }
            return max_route_dist;
        }
    }
    
    __device__ float compute_penalty(const Sol& s) const {
        float penalty = 0.0f;
        for (int v = 0; v < max_vehicles; v++) {
            float load = 0.0f;
            for (int i = 0; i < s.dim2_sizes[v]; i++) {
                load += d_demand[s.data[v][i]];
            }
            if (load > capacity) {
                penalty += (load - capacity) * 100.0f;
            }
        }
        return penalty;
    }
    
    // 运行时配置覆盖
    CompareMode override_mode = CompareMode::Weighted;
    float override_weights[3] = {0.6f, 0.2f, 0.2f};
    int override_priority[3] = {0, 1, 2};
    float override_tolerance[3] = {0.0f, 0.0f, 0.0f};
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = max_vehicles;
        cfg.dim2_default = 0;
        fill_obj_config(cfg);
        cfg.cross_row_prob = 0.3f;
        cfg.row_mode = RowMode::Partition;
        cfg.total_elements = n;
        
        // 应用运行时覆盖
        cfg.compare_mode = override_mode;
        for (int i = 0; i < 3; i++) {
            cfg.obj_weights[i] = override_weights[i];
            cfg.obj_priority[i] = override_priority[i];
            cfg.obj_tolerance[i] = override_tolerance[i];
        }
        
        return cfg;
    }
    
    size_t working_set_bytes() const {
        return (size_t)(n + 1) * (n + 1) * sizeof(float) + (size_t)n * sizeof(float);
    }
    
    static TriObjectiveVRP create(const float* h_dist_matrix, const float* h_demand_array,
                                   int num_customers, float vehicle_capacity, int max_veh) {
        TriObjectiveVRP prob;
        prob.n = num_customers;
        prob.capacity = vehicle_capacity;
        prob.max_vehicles = max_veh;
        
        size_t dist_size = (num_customers + 1) * (num_customers + 1) * sizeof(float);
        size_t demand_size = num_customers * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&prob.d_dist, dist_size));
        CUDA_CHECK(cudaMalloc(&prob.d_demand, demand_size));
        CUDA_CHECK(cudaMemcpy((void*)prob.d_dist, h_dist_matrix, dist_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)prob.d_demand, h_demand_array, demand_size, cudaMemcpyHostToDevice));
        
        return prob;
    }
    
    void destroy() {
        if (d_dist) CUDA_CHECK(cudaFree((void*)d_dist));
        if (d_demand) CUDA_CHECK(cudaFree((void*)d_demand));
    }
    
    TriObjectiveVRP* clone_to_device(int gpu_id) const override {
        int orig_device;
        CUDA_CHECK(cudaGetDevice(&orig_device));
        CUDA_CHECK(cudaSetDevice(gpu_id));
        
        // 在目标 GPU 上分配设备内存
        float* dd;
        float* ddem;
        size_t dist_size = (n + 1) * (n + 1) * sizeof(float);
        size_t demand_size = n * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&dd, dist_size));
        CUDA_CHECK(cudaMalloc(&ddem, demand_size));
        
        // 从原设备读取数据到 host
        float* h_dist = new float[(n+1) * (n+1)];
        float* h_demand = new float[n];
        CUDA_CHECK(cudaSetDevice(orig_device));
        CUDA_CHECK(cudaMemcpy(h_dist, d_dist, dist_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_demand, d_demand, demand_size, cudaMemcpyDeviceToHost));
        
        // 写入目标设备
        CUDA_CHECK(cudaSetDevice(gpu_id));
        CUDA_CHECK(cudaMemcpy(dd, h_dist, dist_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ddem, h_demand, demand_size, cudaMemcpyHostToDevice));
        
        // 恢复原设备
        CUDA_CHECK(cudaSetDevice(orig_device));
        
        // 创建新的 host 端 Problem 实例
        TriObjectiveVRP* new_prob = new TriObjectiveVRP();
        new_prob->n = n;
        new_prob->capacity = capacity;
        new_prob->max_vehicles = max_vehicles;
        new_prob->d_dist = dd;
        new_prob->d_demand = ddem;
        new_prob->override_mode = override_mode;
        for (int i = 0; i < 3; i++) {
            new_prob->override_weights[i] = override_weights[i];
            new_prob->override_priority[i] = override_priority[i];
            new_prob->override_tolerance[i] = override_tolerance[i];
        }
        
        delete[] h_dist;
        delete[] h_demand;
        
        return new_prob;
    }
};

// 类外定义静态成员
constexpr ObjDef TriObjectiveVRP::OBJ_DEFS[];
