#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

/**
 * 双目标 Knapsack: 最大化价值 + 最小化重量
 * 
 * 目标1: 总价值（最大化）
 * 目标2: 总重量（最小化，在满足容量约束下尽量少用重量）
 * 
 * 测试场景：
 * - Weighted 模式：权重配置 [0.8, 0.2]（80% 关注价值）
 * - Lexicographic 模式：优先级 [价值, 重量]
 */
struct BiObjectiveKnapsack : ProblemBase<BiObjectiveKnapsack, 1, 128> {
    const int* d_values;
    const int* d_weights;
    int n;
    int capacity;
    
    // 双目标定义
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Maximize, 1.0f, 0.0f},  // 目标0: 最大化总价值
        {ObjDir::Minimize, 1.0f, 0.0f},  // 目标1: 最小化总重量
    };
    
    __device__ float compute_obj(int obj_idx, const Sol& s) const {
        if (obj_idx == 0) {
            // 目标1: 总价值（最大化）
            int total_value = 0;
            for (int i = 0; i < s.dim2_sizes[0]; i++) {
                if (s.data[0][i] == 1) {
                    total_value += d_values[i];
                }
            }
            return (float)total_value;
        } else {
            // 目标2: 总重量（最小化）
            int total_weight = 0;
            for (int i = 0; i < s.dim2_sizes[0]; i++) {
                if (s.data[0][i] == 1) {
                    total_weight += d_weights[i];
                }
            }
            return (float)total_weight;
        }
    }
    
    __device__ float compute_penalty(const Sol& s) const {
        int total_weight = 0;
        for (int i = 0; i < s.dim2_sizes[0]; i++) {
            if (s.data[0][i] == 1) {
                total_weight += d_weights[i];
            }
        }
        if (total_weight > capacity) {
            return (float)(total_weight - capacity) * 10.0f;
        }
        return 0.0f;
    }
    
    // 运行时配置覆盖
    CompareMode override_mode = CompareMode::Weighted;
    float override_weights[2] = {0.8f, 0.2f};
    int override_priority[2] = {0, 1};
    float override_tolerance[2] = {0.0f, 0.0f};
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Binary;
        cfg.dim1 = 1;
        cfg.dim2_default = n;
        fill_obj_config(cfg);
        
        // 应用运行时覆盖
        cfg.compare_mode = override_mode;
        for (int i = 0; i < 2; i++) {
            cfg.obj_weights[i] = override_weights[i];
            cfg.obj_priority[i] = override_priority[i];
            cfg.obj_tolerance[i] = override_tolerance[i];
        }
        
        return cfg;
    }
    
    size_t working_set_bytes() const {
        return (size_t)n * (sizeof(int) + sizeof(int));
    }
    
    static BiObjectiveKnapsack create(const int* h_values, const int* h_weights,
                                       int num_items, int knapsack_capacity) {
        BiObjectiveKnapsack prob;
        prob.n = num_items;
        prob.capacity = knapsack_capacity;
        
        size_t size = num_items * sizeof(int);
        
        CUDA_CHECK(cudaMalloc(&prob.d_values, size));
        CUDA_CHECK(cudaMalloc(&prob.d_weights, size));
        CUDA_CHECK(cudaMemcpy((void*)prob.d_values, h_values, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)prob.d_weights, h_weights, size, cudaMemcpyHostToDevice));
        
        return prob;
    }
    
    void destroy() {
        if (d_values) CUDA_CHECK(cudaFree((void*)d_values));
        if (d_weights) CUDA_CHECK(cudaFree((void*)d_weights));
    }
    
    BiObjectiveKnapsack* clone_to_device(int gpu_id) const override {
        int orig_device;
        CUDA_CHECK(cudaGetDevice(&orig_device));
        CUDA_CHECK(cudaSetDevice(gpu_id));
        
        // 在目标 GPU 上分配设备内存
        int* dv;
        int* dw;
        size_t size = n * sizeof(int);
        
        CUDA_CHECK(cudaMalloc(&dv, size));
        CUDA_CHECK(cudaMalloc(&dw, size));
        
        // 从原设备读取数据到 host
        int* h_values = new int[n];
        int* h_weights = new int[n];
        CUDA_CHECK(cudaSetDevice(orig_device));
        CUDA_CHECK(cudaMemcpy(h_values, d_values, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_weights, d_weights, size, cudaMemcpyDeviceToHost));
        
        // 写入目标设备
        CUDA_CHECK(cudaSetDevice(gpu_id));
        CUDA_CHECK(cudaMemcpy(dv, h_values, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dw, h_weights, size, cudaMemcpyHostToDevice));
        
        // 恢复原设备
        CUDA_CHECK(cudaSetDevice(orig_device));
        
        // 创建新的 host 端 Problem 实例
        BiObjectiveKnapsack* new_prob = new BiObjectiveKnapsack();
        new_prob->n = n;
        new_prob->capacity = capacity;
        new_prob->d_values = dv;
        new_prob->d_weights = dw;
        new_prob->override_mode = override_mode;
        for (int i = 0; i < 2; i++) {
            new_prob->override_weights[i] = override_weights[i];
            new_prob->override_priority[i] = override_priority[i];
            new_prob->override_tolerance[i] = override_tolerance[i];
        }
        
        delete[] h_values;
        delete[] h_weights;
        
        return new_prob;
    }
};

// 类外定义静态成员
constexpr ObjDef BiObjectiveKnapsack::OBJ_DEFS[];
