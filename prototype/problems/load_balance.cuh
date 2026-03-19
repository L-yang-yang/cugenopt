/**
 * load_balance.cuh - 离散负载均衡问题（Integer 编码验证）
 * 
 * N 个任务分配到 M 台机器，每个任务有一个处理时间 p[i]。
 * 决策变量：data[0][i] ∈ [0, M-1]，表示任务 i 分配到哪台机器。
 * 目标：最小化 makespan（最大机器负载）。
 * 
 * 已知 NP-hard（等价于 multiprocessor scheduling / load balancing）。
 * LPT（最长处理时间优先）贪心可得 4/3 近似。
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

struct LoadBalanceProblem : ProblemBase<LoadBalanceProblem, 1, 64> {
    const float* d_proc_time;   // 任务处理时间 [N]
    int n;                      // 任务数
    int m;                      // 机器数
    
    __device__ float calc_makespan(const Sol& sol) const {
        float load[32] = {};    // 最多 32 台机器
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++) {
            int machine = sol.data[0][i];
            if (machine >= 0 && machine < m)
                load[machine] += d_proc_time[i];
        }
        float max_load = 0.0f;
        for (int j = 0; j < m; j++)
            if (load[j] > max_load) max_load = load[j];
        return max_load;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},   // case 0: makespan
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_makespan(sol);
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        return 0.0f;   // 无约束（任何分配都合法）
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Integer;
        cfg.dim1 = 1;  cfg.dim2_default = n;
        cfg.value_lower_bound = 0;
        cfg.value_upper_bound = m - 1;
        fill_obj_config(cfg);
        return cfg;
    }
    
    size_t shared_mem_bytes() const {
        return (size_t)n * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sp = reinterpret_cast<float*>(smem);
        for (int i = tid; i < n; i += bsz) sp[i] = d_proc_time[i];
        d_proc_time = sp;
    }
    
    static LoadBalanceProblem create(const float* h_proc_time, int n, int m) {
        LoadBalanceProblem prob;
        prob.n = n; prob.m = m;
        float* dp;
        CUDA_CHECK(cudaMalloc(&dp, sizeof(float) * n));
        CUDA_CHECK(cudaMemcpy(dp, h_proc_time, sizeof(float) * n, cudaMemcpyHostToDevice));
        prob.d_proc_time = dp;
        return prob;
    }
    
    void destroy() {
        if (d_proc_time) cudaFree(const_cast<float*>(d_proc_time));
        d_proc_time = nullptr;
    }
};
