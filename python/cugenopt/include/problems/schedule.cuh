/**
 * schedule.cuh - 排班问题
 * 
 * 继承 ProblemBase，使用 ObjDef 目标注册机制
 * 2 个目标：总成本（min）+ 不公平度（min，权重更高）
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

struct ScheduleProblem : ProblemBase<ScheduleProblem, 8, 16> {
    const float* d_cost;
    int days, emps, required;
    
    // ---- 目标计算 ----
    __device__ float calc_total_cost(const Sol& sol) const {
        float total = 0.0f;
        for (int d = 0; d < days; d++)
            for (int e = 0; e < emps; e++)
                if (sol.data[d][e]) total += d_cost[d * emps + e];
        return total;
    }
    
    __device__ float calc_unfairness(const Sol& sol) const {
        int workdays[D2];
        for (int e = 0; e < emps; e++) workdays[e] = 0;
        for (int d = 0; d < days; d++)
            for (int e = 0; e < emps; e++)
                if (sol.data[d][e]) workdays[e]++;
        int max_w = 0, min_w = days;
        for (int e = 0; e < emps; e++) {
            if (workdays[e] > max_w) max_w = workdays[e];
            if (workdays[e] < min_w) min_w = workdays[e];
        }
        return (float)(max_w - min_w);
    }
    
    // ---- 目标定义（OBJ_DEFS 与 compute_obj 必须一一对应）----
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},   // case 0: calc_total_cost
        {ObjDir::Minimize, 5.0f, 0.0f},   // case 1: calc_unfairness
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_total_cost(sol);     // OBJ_DEFS[0]
            case 1: return calc_unfairness(sol);     // OBJ_DEFS[1]
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        float penalty = 0.0f;
        for (int d = 0; d < days; d++) {
            int count = 0;
            for (int e = 0; e < emps; e++)
                if (sol.data[d][e]) count++;
            int diff = count - required;
            penalty += (diff > 0) ? (float)diff : (float)(-diff);
        }
        return penalty;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Binary;
        cfg.dim1 = days;  cfg.dim2_default = emps;
        cfg.row_mode = RowMode::Fixed;
        fill_obj_config(cfg);
        return cfg;
    }
    
    // 默认回退全量（基类行为）— 不需要覆盖 evaluate_move
    
    // ---- shared memory 接口 ----
    size_t shared_mem_bytes() const {
        return (size_t)days * emps * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sc = reinterpret_cast<float*>(smem);
        int total = days * emps;
        for (int i = tid; i < total; i += bsz) sc[i] = d_cost[i];
        d_cost = sc;
    }
    
    static ScheduleProblem create(const float* hc, int days, int emps, int req) {
        ScheduleProblem prob;
        prob.days = days; prob.emps = emps; prob.required = req;
        float* dc;
        CUDA_CHECK(cudaMalloc(&dc, sizeof(float)*days*emps));
        CUDA_CHECK(cudaMemcpy(dc, hc, sizeof(float)*days*emps, cudaMemcpyHostToDevice));
        prob.d_cost = dc;
        return prob;
    }
    
    void destroy() {
        if (d_cost) { cudaFree(const_cast<float*>(d_cost)); d_cost = nullptr; }
    }
};
