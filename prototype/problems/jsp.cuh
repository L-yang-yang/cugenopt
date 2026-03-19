/**
 * jsp.cuh - 车间调度问题 (Job Shop Scheduling Problem)
 * 
 * J 个工件，每个工件有 O 道工序，每道工序指定机器和耗时。
 * 
 * === 编码方案 A：Integer 多行（时间表编码）===
 * JSPProblem: data[j][i] = 工件 j 的第 i 道工序的开始时间
 *   dim1 = num_jobs, dim2_default = num_ops
 *   row_mode = Fixed（禁止 ROW_SPLIT/ROW_MERGE）
 *   每行代表一个工件的固定工序序列，行长度不可变
 * 
 * === 编码方案 B：Permutation 多重集（工序排列编码）===
 * JSPPermProblem: data[0][k] = 工件编号（0..J-1），长度 J*O
 *   值 j 出现 O 次。从左到右扫描，第 t 次遇到值 j 表示工件 j 的第 t 道工序。
 *   dim1 = 1, dim2_default = J*O, perm_repeat_count = O
 *   标准 Permutation 算子（swap/reverse/insert）天然保持多重集结构
 * 
 * 目标：Minimize makespan（所有工件完成时间的最大值）。
 * 约束：
 *   (a) 工序顺序：同一工件的工序必须按序执行
 *   (b) 机器冲突：同一机器同一时刻只能处理一个工序
 * 
 * 验证实例：自定义 3 工件 3 机器 (3x3)，最优 makespan = 12
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

// ============================================================
// 编码方案 A：Integer 多行（时间表编码）
// ============================================================

struct JSPProblem : ProblemBase<JSPProblem, 8, 16> {
    const int*   d_machine;     // 工序所需机器 [J*O]
    const float* d_duration;    // 工序耗时 [J*O]
    int num_jobs;               // 工件数 J
    int num_ops;                // 每工件工序数 O
    int num_machines;           // 机器数 M
    int time_horizon;           // 时间上界
    
    __device__ float calc_makespan(const Sol& sol) const {
        float makespan = 0.0f;
        for (int j = 0; j < num_jobs; j++) {
            int last = num_ops - 1;
            float end = (float)sol.data[j][last] + d_duration[j * num_ops + last];
            if (end > makespan) makespan = end;
        }
        return makespan;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_makespan(sol);
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        float penalty = 0.0f;
        
        // (a) 工序顺序约束
        for (int j = 0; j < num_jobs; j++) {
            for (int i = 1; i < num_ops; i++) {
                float prev_end = (float)sol.data[j][i-1] + d_duration[j * num_ops + (i-1)];
                float curr_start = (float)sol.data[j][i];
                if (curr_start < prev_end)
                    penalty += (prev_end - curr_start) * 10.0f;
            }
        }
        
        // (b) 机器冲突约束
        int total = num_jobs * num_ops;
        for (int a = 0; a < total; a++) {
            int ja = a / num_ops, ia = a % num_ops;
            int m_a = d_machine[a];
            float s_a = (float)sol.data[ja][ia];
            float e_a = s_a + d_duration[a];
            for (int b = a + 1; b < total; b++) {
                if (d_machine[b] != m_a) continue;
                int jb = b / num_ops, ib = b % num_ops;
                float s_b = (float)sol.data[jb][ib];
                float e_b = s_b + d_duration[b];
                float overlap = fminf(e_a, e_b) - fmaxf(s_a, s_b);
                if (overlap > 0.0f)
                    penalty += overlap * 10.0f;
            }
        }
        
        return penalty;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Integer;
        cfg.dim1 = num_jobs;
        cfg.dim2_default = num_ops;
        cfg.value_lower_bound = 0;
        cfg.value_upper_bound = time_horizon - 1;
        cfg.row_mode = RowMode::Fixed;
        fill_obj_config(cfg);
        return cfg;
    }
    
    size_t shared_mem_bytes() const {
        int total = num_jobs * num_ops;
        return (size_t)total * (sizeof(int) + sizeof(float));
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        int total = num_jobs * num_ops;
        int* sm = reinterpret_cast<int*>(smem);
        for (int i = tid; i < total; i += bsz) sm[i] = d_machine[i];
        d_machine = sm;
        
        float* sd = reinterpret_cast<float*>(sm + total);
        for (int i = tid; i < total; i += bsz) sd[i] = d_duration[i];
        d_duration = sd;
    }
    
    static JSPProblem create(const int* h_machine, const float* h_duration,
                              int num_jobs, int num_ops, int num_machines,
                              int time_horizon) {
        JSPProblem prob;
        prob.num_jobs = num_jobs;
        prob.num_ops = num_ops;
        prob.num_machines = num_machines;
        prob.time_horizon = time_horizon;
        
        int total = num_jobs * num_ops;
        int* dm;
        CUDA_CHECK(cudaMalloc(&dm, sizeof(int) * total));
        CUDA_CHECK(cudaMemcpy(dm, h_machine, sizeof(int) * total, cudaMemcpyHostToDevice));
        prob.d_machine = dm;
        
        float* dd;
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * total));
        CUDA_CHECK(cudaMemcpy(dd, h_duration, sizeof(float) * total, cudaMemcpyHostToDevice));
        prob.d_duration = dd;
        
        return prob;
    }
    
    void destroy() {
        if (d_machine)  { cudaFree(const_cast<int*>(d_machine));     d_machine = nullptr; }
        if (d_duration) { cudaFree(const_cast<float*>(d_duration));  d_duration = nullptr; }
    }
};

// ============================================================
// 编码方案 B：Permutation 多重集（工序排列编码）
// ============================================================
// data[0] 是长度 J*O 的排列，值域 [0, J)，每个值出现 O 次
// 从左到右扫描：第 t 次遇到值 j → 安排工件 j 的第 t 道工序
// 贪心解码：每道工序安排在"最早可行时间"（满足工序顺序 + 机器空闲）

struct JSPPermProblem : ProblemBase<JSPPermProblem, 1, 64> {
    const int*   d_machine;     // 工序所需机器 [J*O]
    const float* d_duration;    // 工序耗时 [J*O]
    int num_jobs;
    int num_ops;
    int num_machines;
    
    // 贪心解码：从排列生成调度方案，返回 makespan
    __device__ float decode_and_makespan(const Sol& sol) const {
        int total = num_jobs * num_ops;
        int size = sol.dim2_sizes[0];
        if (size < total) return 1e9f;
        
        float job_avail[8];     // 每个工件的下一道工序最早开始时间
        float mach_avail[8];    // 每台机器的最早空闲时间
        int   job_next_op[8];   // 每个工件的下一道待安排工序编号
        
        for (int j = 0; j < num_jobs; j++) { job_avail[j] = 0.0f; job_next_op[j] = 0; }
        for (int m = 0; m < num_machines; m++) mach_avail[m] = 0.0f;
        
        float makespan = 0.0f;
        for (int k = 0; k < total; k++) {
            int j = sol.data[0][k];
            if (j < 0 || j >= num_jobs) return 1e9f;
            int op = job_next_op[j];
            if (op >= num_ops) continue;  // 该工件已安排完
            
            int flat = j * num_ops + op;
            int m = d_machine[flat];
            float dur = d_duration[flat];
            
            // 最早开始时间 = max(工件前序完成, 机器空闲)
            float start = fmaxf(job_avail[j], mach_avail[m]);
            float end = start + dur;
            
            job_avail[j] = end;
            mach_avail[m] = end;
            job_next_op[j] = op + 1;
            
            if (end > makespan) makespan = end;
        }
        
        return makespan;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return decode_and_makespan(sol);
            default: return 0.0f;
        }
    }
    
    // 贪心解码天然满足约束，penalty 始终为 0
    __device__ float compute_penalty(const Sol& sol) const {
        return 0.0f;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = 1;
        cfg.dim2_default = num_jobs * num_ops;
        cfg.perm_repeat_count = num_ops;
        fill_obj_config(cfg);
        return cfg;
    }
    
    size_t shared_mem_bytes() const {
        int total = num_jobs * num_ops;
        return (size_t)total * (sizeof(int) + sizeof(float));
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        int total = num_jobs * num_ops;
        int* sm = reinterpret_cast<int*>(smem);
        for (int i = tid; i < total; i += bsz) sm[i] = d_machine[i];
        d_machine = sm;
        
        float* sd = reinterpret_cast<float*>(sm + total);
        for (int i = tid; i < total; i += bsz) sd[i] = d_duration[i];
        d_duration = sd;
    }
    
    static JSPPermProblem create(const int* h_machine, const float* h_duration,
                                  int num_jobs, int num_ops, int num_machines) {
        JSPPermProblem prob;
        prob.num_jobs = num_jobs;
        prob.num_ops = num_ops;
        prob.num_machines = num_machines;
        
        int total = num_jobs * num_ops;
        int* dm;
        CUDA_CHECK(cudaMalloc(&dm, sizeof(int) * total));
        CUDA_CHECK(cudaMemcpy(dm, h_machine, sizeof(int) * total, cudaMemcpyHostToDevice));
        prob.d_machine = dm;
        
        float* dd;
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * total));
        CUDA_CHECK(cudaMemcpy(dd, h_duration, sizeof(float) * total, cudaMemcpyHostToDevice));
        prob.d_duration = dd;
        
        return prob;
    }
    
    void destroy() {
        if (d_machine)  { cudaFree(const_cast<int*>(d_machine));     d_machine = nullptr; }
        if (d_duration) { cudaFree(const_cast<float*>(d_duration));  d_duration = nullptr; }
    }
};
