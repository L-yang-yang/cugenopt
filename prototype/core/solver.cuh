/**
 * solver.cuh - 主求解循环
 * 
 * v2.0: Block 级架构重构
 *   - 1 block = 1 solution（邻域并行）
 *   - Solution 存放在 shared memory
 *   - 每代：K 个线程各自生成候选 move + 评估 delta → 归约选最优 → thread 0 执行
 *   - 交叉暂用简化版（thread 0 执行，其余线程等待）
 *   - 迁移/精英注入保持单线程 kernel（操作全局内存）
 *
 * 要求 Problem 接口：
 *   size_t shared_mem_bytes() const;
 *   __device__ void load_shared(char* smem, int tid, int bsz);
 *   __device__ void evaluate(Sol& sol) const;
 */

#pragma once
#include "types.cuh"
#include "population.cuh"
#include "operators.cuh"
#include "relation_matrix.cuh"
#include "cuda_utils.cuh"
#include "init_selection.cuh"
#include "init_heuristic.cuh"
#include <cmath>

// ============================================================
// 编译时常量
// ============================================================
constexpr int BLOCK_LEVEL_THREADS = 128;  // Block 级架构的默认线程数/block

// ============================================================
// EvolveParams — CUDA Graph 可变参数（device memory）
// ============================================================
// 将每个 batch 会变化的参数集中到一个 struct 中，
// evolve_block_kernel 通过指针读取，CUDA Graph 录制时绑定指针。
// 每次 replay 前只需 cudaMemcpy 更新这块 device memory。

struct EvolveParams {
    float       temp_start;
    int         gens_per_batch;
    SeqRegistry seq_reg;
    KStepConfig kstep;
    int         migrate_round;
    ObjConfig   oc;
};

// ============================================================
// 工具：协作加载/存储 Solution（shared memory ↔ global memory）
// ============================================================

template<typename Sol>
__device__ inline void cooperative_load_sol(Sol& dst, const Sol& src,
                                             int tid, int num_threads) {
    // 按 int 粒度协作拷贝整个 Solution 结构体
    const int* src_ptr = reinterpret_cast<const int*>(&src);
    int* dst_ptr = reinterpret_cast<int*>(&dst);
    constexpr int n_ints = (sizeof(Sol) + sizeof(int) - 1) / sizeof(int);
    for (int i = tid; i < n_ints; i += num_threads)
        dst_ptr[i] = src_ptr[i];
}

template<typename Sol>
__device__ inline void cooperative_store_sol(Sol& dst, const Sol& src,
                                              int tid, int num_threads) {
    cooperative_load_sol(dst, src, tid, num_threads);  // 同样的拷贝逻辑
}

// ============================================================
// Kernel 1: 初始评估（只调用一次，1 block = 1 solution）
// ============================================================

template<typename Problem, typename Sol>
__global__ void evaluate_kernel(Problem prob, Sol* pop, int pop_size,
                                 size_t smem_size) {
    extern __shared__ char smem[];
    Problem lp = prob;
    if (smem_size > 0) { lp.load_shared(smem, threadIdx.x, blockDim.x); __syncthreads(); }
    
    // 1-thread-per-solution 初始评估（保持简单，只调用一次）
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pop_size) lp.evaluate(pop[tid]);
}

// ============================================================
// Kernel 2: Block 级批量进化（邻域并行）
// ============================================================
//
// 每代流程：
//   1. K 个线程各自生成一个候选 move
//   2. K 个线程各自评估 move 的 delta（不修改 shared memory 中的 sol）
//   3. Block 内归约：选 delta 最小的 move
//   4. Thread 0 决定是否接受（SA / HC）
//   5. Thread 0 执行最优 move 并更新 sol
//   6. __syncthreads() 让所有线程看到更新后的 sol
//
// Solution 在 shared memory 中，Problem 数据也在 shared memory 中

// ============================================================
// MultiStepCandidate — 多步执行结果（用于归约）
// ============================================================
struct MultiStepCandidate {
    float delta;
    float new_penalty;
    int   seq_indices[MAX_K];
    int   k_steps;
    int   winner_tid;
};

template<typename Problem, typename Sol>
__global__ void evolve_block_kernel(Problem prob, Sol* pop, int pop_size,
                                     EncodingType encoding, int dim1,
                                     ObjConfig oc_legacy,
                                     curandState* rng_states,
                                     float alpha,
                                     size_t prob_smem_size,
                                     AOSStats* d_aos_stats,
                                     const float* d_G,
                                     const float* d_O,
                                     int rel_N,
                                     int val_lb,
                                     int val_ub,
                                     const EvolveParams* d_params) {
    extern __shared__ char smem[];
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    if (bid >= pop_size) return;
    
    const int gens_per_batch = d_params->gens_per_batch;
    const SeqRegistry seq_reg = d_params->seq_reg;
    const KStepConfig kstep = d_params->kstep;
    const float temp_start = d_params->temp_start;
    const ObjConfig oc = d_params->oc;
    
    // --- shared memory 布局 ---
    // [0 .. sizeof(Sol)-1]                              : Solution
    // [sizeof(Sol) .. sizeof(Sol)+prob_smem-1]          : Problem 数据
    // [之后 .. ]                                        : MultiStepCandidate[num_threads] 归约工作区
    // [之后 .. ]                                        : AOSStats (如果启用)
    
    Sol* s_sol = reinterpret_cast<Sol*>(smem);
    char* prob_smem_ptr = smem + sizeof(Sol);
    MultiStepCandidate* s_cands = reinterpret_cast<MultiStepCandidate*>(
        smem + sizeof(Sol) + prob_smem_size);
    
    // AOS 统计（在 MultiStepCandidate 数组之后）
    AOSStats* s_aos = nullptr;
    if (d_aos_stats) {
        s_aos = reinterpret_cast<AOSStats*>(
            smem + sizeof(Sol) + prob_smem_size + sizeof(MultiStepCandidate) * num_threads);
        // Thread 0 初始化 AOS 计数器
        if (tid == 0) {
            for (int i = 0; i < MAX_SEQ; i++) {
                s_aos->usage[i] = 0;
                s_aos->improvement[i] = 0;
            }
            for (int i = 0; i < MAX_K; i++) {
                s_aos->k_usage[i] = 0;
                s_aos->k_improvement[i] = 0;
            }
        }
    }
    
    // 加载 Problem 数据到 shared memory
    Problem lp = prob;
    if (prob_smem_size > 0) {
        lp.load_shared(prob_smem_ptr, tid, num_threads);
    }
    
    // 协作加载 Solution 到 shared memory
    cooperative_load_sol(*s_sol, pop[bid], tid, num_threads);
    __syncthreads();
    
    int rng_idx = bid * num_threads + tid;
    curandState rng = rng_states[rng_idx];
    
    float temp = temp_start;
    
    for (int g = 0; g < gens_per_batch; g++) {
        // ============================================================
        // Step 1: 每个线程独立采样 K 步数 + K 个序列，在 local copy 上执行
        // ============================================================
        
        // 采样 K（步数）：按 kstep.weights 权重
        float kr = curand_uniform(&rng);
        int my_k = 1;  // 默认 K=1
        {
            float cum = 0.0f;
            for (int i = 0; i < MAX_K; i++) {
                cum += kstep.weights[i];
                if (kr < cum) { my_k = i + 1; break; }
            }
        }
        
        // 在 local memory 拷贝 sol，执行 K 步 move
        Sol local_sol = *s_sol;
        MultiStepCandidate my_cand;
        my_cand.k_steps = my_k;
        my_cand.winner_tid = tid;
        for (int i = 0; i < MAX_K; i++) {
            my_cand.seq_indices[i] = -1;
        }
        
        bool all_noop = true;
        for (int step = 0; step < my_k; step++) {
            int seq_idx = -1;
            bool changed = ops::sample_and_execute(
                seq_reg, local_sol, dim1, encoding, &rng, seq_idx,
                d_G, d_O, rel_N, val_lb, val_ub,
                static_cast<const void*>(&lp));
            my_cand.seq_indices[step] = seq_idx;
            if (changed) all_noop = false;
        }
        
        // Step 2: 评估最终 delta（K 步之后 vs 原始 sol）
        if (all_noop) {
            my_cand.delta = 1e30f;
            my_cand.new_penalty = s_sol->penalty;
        } else {
            lp.evaluate(local_sol);
            float old_scalar = obj_scalar(s_sol->objectives, oc);
            float new_scalar = obj_scalar(local_sol.objectives, oc);
            
            bool old_feasible = (s_sol->penalty <= 0.0f);
            bool new_feasible = (local_sol.penalty <= 0.0f);
            
            if (new_feasible && !old_feasible) {
                my_cand.delta = -1e20f;
            } else if (!new_feasible && old_feasible) {
                my_cand.delta = 1e20f;
            } else if (!new_feasible && !old_feasible) {
                my_cand.delta = local_sol.penalty - s_sol->penalty;
            } else {
                my_cand.delta = new_scalar - old_scalar;
            }
            my_cand.new_penalty = local_sol.penalty;
        }
        
        s_cands[tid] = my_cand;
        __syncthreads();
        
        // Step 3: Block 内并行归约，找 delta 最小的 candidate
        for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_cands[tid + stride].delta < s_cands[tid].delta)
                    s_cands[tid] = s_cands[tid + stride];
            }
            __syncthreads();
        }
        
        // Step 4: Thread 0 决定是否接受
        if (tid == 0) {
            MultiStepCandidate& best = s_cands[0];
            bool has_valid = (best.delta < 1e29f);
            
            if (has_valid) {
                bool improved = (best.delta < 0.0f);
                
                bool accept;
                if (improved) {
                    accept = true;
                } else if (temp > 0.0f && s_sol->penalty <= 0.0f && best.new_penalty <= 0.0f) {
                    accept = curand_uniform(&rng) < expf(-best.delta / temp);
                } else {
                    accept = false;
                }
                
                if (accept) {
                    // AOS 统计：K 层 + 算子层
                    if (s_aos) {
                        int ki = best.k_steps - 1;
                        if (ki >= 0 && ki < MAX_K) {
                            s_aos->k_usage[ki]++;
                            if (improved) s_aos->k_improvement[ki]++;
                        }
                        for (int step = 0; step < best.k_steps; step++) {
                            int si = best.seq_indices[step];
                            if (si >= 0 && si < seq_reg.count) {
                                s_aos->usage[si]++;
                                if (improved) s_aos->improvement[si]++;
                            }
                        }
                    }
                    // Signal: keep winner_tid as-is (accept)
                } else {
                    s_cands[0].winner_tid = -1;  // Signal: reject
                }
            } else {
                s_cands[0].winner_tid = -1;  // Signal: no valid candidate
            }
            
            temp *= alpha;
        }
        __syncthreads();
        
        // Step 5: Winner thread writes local_sol to s_sol
        int winner = s_cands[0].winner_tid;
        if (winner >= 0 && tid == winner) {
            *s_sol = local_sol;
        }
        __syncthreads();
    }
    
    // 写回 Solution 到全局内存
    cooperative_store_sol(pop[bid], *s_sol, tid, num_threads);
    
    // AOS 统计写回全局内存
    if (d_aos_stats && tid == 0) {
        d_aos_stats[bid] = *s_aos;
    }
    
    // 保存 RNG 状态
    rng_states[rng_idx] = rng;
}

// ============================================================
// Kernel 2b: Block 级交叉操作
// ============================================================
// 简化版：thread 0 执行交叉逻辑，其余线程协作加载/存储
// 后续 Phase 3 会实现多线程协作交叉

template<typename Problem, typename Sol>
__global__ void crossover_block_kernel(Problem prob, Sol* pop, int pop_size,
                                        EncodingType encoding, int dim1,
                                        ObjConfig oc,
                                        curandState* rng_states,
                                        float crossover_rate,
                                        size_t prob_smem_size,
                                        int total_elements = 0) {
    extern __shared__ char smem[];
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int K = blockDim.x;
    
    if (bid >= pop_size) return;
    
    // shared memory 布局：Sol + Problem data
    Sol* s_sol = reinterpret_cast<Sol*>(smem);
    char* prob_smem_ptr = smem + sizeof(Sol);
    
    Problem lp = prob;
    if (prob_smem_size > 0) {
        lp.load_shared(prob_smem_ptr, tid, K);
    }
    
    cooperative_load_sol(*s_sol, pop[bid], tid, K);
    __syncthreads();
    
    // Thread 0 执行交叉逻辑
    if (tid == 0) {
        int rng_idx = bid * K;
        curandState rng = rng_states[rng_idx];
        
        if (curand_uniform(&rng) < crossover_rate) {
            int c1 = rand_int(&rng, pop_size);
            int c2 = rand_int(&rng, pop_size - 1);
            if (c2 >= c1) c2++;
            int mate_idx = is_better(pop[c1], pop[c2], oc) ? c1 : c2;
            
            if (mate_idx != bid) {
                const Sol& mate = pop[mate_idx];
                Sol child;
                bool did_crossover = false;
                
                if (encoding == EncodingType::Permutation) {
                    int te = total_elements;
                    if (te <= 0) te = s_sol->dim2_sizes[0];
                    ops::perm_ox_crossover(child, *s_sol, mate, dim1, te, &rng);
                    did_crossover = true;
                } else if (encoding == EncodingType::Binary) {
                    ops::uniform_crossover(child, *s_sol, mate, dim1, &rng);
                    did_crossover = true;
                }
                
                if (did_crossover) {
                    lp.evaluate(child);
                    if (is_better(child, *s_sol, oc)) {
                        *s_sol = child;
                    }
                }
            }
        }
        
        rng_states[rng_idx] = rng;
    }
    __syncthreads();
    
    // 写回（可能被交叉更新了）
    cooperative_store_sol(pop[bid], *s_sol, tid, K);
}

// ============================================================
// Kernel 3: 岛屿间迁移（保持不变，单线程 kernel）
// ============================================================

template<typename Sol>
__device__ inline int find_worst_in_island(const Sol* pop, int base, int island_size,
                                            const ObjConfig& oc) {
    int worst = base;
    for (int i = base + 1; i < base + island_size; i++)
        if (is_better(pop[worst], pop[i], oc)) worst = i;
    return worst;
}

template<typename Sol>
__global__ void migrate_kernel(Sol* pop, int pop_size, int island_size,
                                ObjConfig oc,
                                MigrateStrategy strategy,
                                const EvolveParams* d_params) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int round = d_params->migrate_round;
    int num_islands = pop_size / island_size;
    
    int candidates[64];
    for (int isle = 0; isle < num_islands; isle++) {
        int base = isle * island_size;
        int best = base;
        for (int i = base + 1; i < base + island_size; i++)
            if (is_better(pop[i], pop[best], oc)) best = i;
        candidates[isle] = best;
    }
    
    int topn[64];
    if (strategy == MigrateStrategy::TopN || strategy == MigrateStrategy::Hybrid) {
        bool selected[64] = {};
        for (int t = 0; t < num_islands; t++) {
            int best_c = -1;
            for (int c = 0; c < num_islands; c++) {
                if (selected[c]) continue;
                if (best_c < 0 || is_better(pop[candidates[c]], pop[candidates[best_c]], oc))
                    best_c = c;
            }
            topn[t] = candidates[best_c];
            selected[best_c] = true;
        }
        for (int i = 0; i < num_islands; i++) {
            int dst_isle = (i + round) % num_islands;
            int dst_base = dst_isle * island_size;
            int worst = find_worst_in_island(pop, dst_base, island_size, oc);
            if (is_better(pop[topn[i]], pop[worst], oc))
                pop[worst] = pop[topn[i]];
        }
    }
    
    if (strategy == MigrateStrategy::Ring || strategy == MigrateStrategy::Hybrid) {
        for (int isle = 0; isle < num_islands; isle++) {
            int dst_isle = (isle + 1) % num_islands;
            int dst_base = dst_isle * island_size;
            int worst = find_worst_in_island(pop, dst_base, island_size, oc);
            int src = candidates[isle];
            if (is_better(pop[src], pop[worst], oc))
                pop[worst] = pop[src];
        }
    }
}

// ============================================================
// Kernel 4: 精英注入（保持不变）
// ============================================================

template<typename Sol>
__global__ void elite_inject_kernel(Sol* pop, int pop_size,
                                     Sol* global_best, ObjConfig oc) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    int best_idx = 0;
    for (int i = 1; i < pop_size; i++)
        if (is_better(pop[i], pop[best_idx], oc)) best_idx = i;
    
    if (is_better(pop[best_idx], *global_best, oc))
        *global_best = pop[best_idx];
    
    int worst_idx = 0;
    for (int i = 1; i < pop_size; i++)
        if (is_better(pop[worst_idx], pop[i], oc)) worst_idx = i;
    
    if (is_better(*global_best, pop[worst_idx], oc))
        pop[worst_idx] = *global_best;
}

// ============================================================
// v5.0: 多 GPU 协同 — 注入外部解到岛屿
// ============================================================

template<typename Sol>
__global__ void inject_to_islands_kernel(Sol* pop, int pop_size, int island_size,
                                          const Sol* inject_solutions, int num_inject,
                                          MultiGpuInjectMode mode, ObjConfig oc) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (num_inject == 0) return;
    
    int num_islands = pop_size / island_size;
    if (num_islands == 0) return;
    
    // 根据注入模式确定注入的岛屿数量
    int islands_to_inject = 0;
    if (mode == MultiGpuInjectMode::OneIsland) {
        islands_to_inject = 1;
    } else if (mode == MultiGpuInjectMode::HalfIslands) {
        islands_to_inject = (num_islands + 1) / 2;
    } else {  // AllIslands
        islands_to_inject = num_islands;
    }
    
    // 将注入解分配到各个岛屿的 worst 位置
    for (int i = 0; i < islands_to_inject && i < num_inject; i++) {
        int target_isle = i % num_islands;
        int base = target_isle * island_size;
        
        // 找到该岛的 worst 解
        int worst = find_worst_in_island(pop, base, island_size, oc);
        
        // 如果注入解更优，则替换
        if (is_better(inject_solutions[i], pop[worst], oc)) {
            pop[worst] = inject_solutions[i];
        }
    }
}

// ============================================================
// v5.0 方案 B3: inject_check_kernel — 被动注入检查
// ============================================================
// GPU 在 migrate 时检查 InjectBuffer，如果有新解则注入到第一个岛的 worst
// 使用 atomicExch 原子读取并清除 flag，确保线程安全
//
// 设计要点：
// 1. 单线程执行（thread 0 of block 0），避免竞争
// 2. atomicExch 原子读取 flag 并清零，确保每个解只被处理一次
// 3. 只注入到第一个岛（OneIsland 策略），保持多样性
// 4. 完全可选：如果 inject_buf 为 nullptr，直接跳过（不影响单 GPU）

template<typename Sol>
__global__ void inject_check_kernel(Sol* pop, int pop_size, int island_size,
                                     InjectBuffer<Sol>* inject_buf, ObjConfig oc) {
    // 单线程执行
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // 如果没有注入缓冲区，直接返回（单 GPU 场景）
    if (inject_buf == nullptr) return;
    
    // 原子读取并清除 flag（确保每个解只被处理一次）
    int flag = atomicExch(inject_buf->d_flag, 0);
    
    // 如果没有新解，直接返回
    if (flag != 1) return;
    
    // 读取注入的解
    Sol inject_sol = *(inject_buf->d_solution);
    
    // 找到第一个岛的 worst 位置
    int num_islands = pop_size / island_size;
    if (num_islands == 0) return;
    
    int worst = find_worst_in_island(pop, 0, island_size, oc);
    
    // 如果注入解更优，则替换
    if (is_better(inject_sol, pop[worst], oc)) {
        pop[worst] = inject_sol;
    }
}

// ============================================================
// solve<Problem>: 主循环（Block 级架构）
// ============================================================

using RegistryCallback = void(*)(SeqRegistry&);

template<typename Problem>
SolveResult<typename Problem::Sol> solve(Problem& prob, const SolverConfig& cfg,
                                          const typename Problem::Sol* init_solutions = nullptr,
                                          int num_init_solutions = 0,
                                          RegistryCallback custom_registry_fn = nullptr,
                                          InjectBuffer<typename Problem::Sol>* inject_buf = nullptr,
                                          typename Problem::Sol** d_global_best_out = nullptr) {
    using Sol = typename Problem::Sol;
    ProblemConfig pcfg = prob.config();
    SolveResult<Sol> result;
    
    bool use_sa = cfg.sa_temp_init > 0.0f;
    bool use_crossover = cfg.crossover_rate > 0.0f;
    bool use_aos = cfg.use_aos;
    bool use_time_limit = cfg.time_limit_sec > 0.0f;
    bool use_stagnation = cfg.stagnation_limit > 0;
    
    // Block 级参数
    const int block_threads = BLOCK_LEVEL_THREADS;  // 128 线程/block
    
    // --- 0. Shared memory 计算（需要在 pop_size 确定之前完成，用于 occupancy 查询）---
    size_t prob_smem = prob.shared_mem_bytes();
    // v3.1: 归约工作区为 MultiStepCandidate（含 K 步 moves + seq_indices）
    size_t total_smem = sizeof(Sol) + prob_smem + sizeof(MultiStepCandidate) * block_threads;
    if (use_aos) total_smem += sizeof(AOSStats);
    
    // 查询 GPU 硬件属性
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // 尝试扩展 shared memory 上限（V100: 96KB, A100: 164KB 等）
    size_t max_smem = (size_t)prop.sharedMemPerBlock;
    if (total_smem > 48 * 1024) {
        cudaError_t err1 = cudaFuncSetAttribute(
            evolve_block_kernel<Problem, Sol>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)total_smem);
        cudaError_t err2 = cudaFuncSetAttribute(
            crossover_block_kernel<Problem, Sol>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)total_smem);
        if (err1 == cudaSuccess && err2 == cudaSuccess) {
            max_smem = total_smem;
        }
    }
    
    // 检查 shared memory 上限
    bool smem_overflow = false;
    if (total_smem > max_smem) {
        smem_overflow = (prob_smem > 0);
        prob_smem = 0;
        total_smem = sizeof(Sol) + sizeof(MultiStepCandidate) * block_threads;
        if (use_aos) total_smem += sizeof(AOSStats);
    }
    
    // --- 0b. 确定 pop_size（自动或用户指定）---
    int pop_size = cfg.pop_size;
    bool auto_pop = (pop_size <= 0);
    
    if (auto_pop) {
        // 查询 occupancy：每个 SM 能同时运行多少个 block
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            evolve_block_kernel<Problem, Sol>,
            block_threads,
            total_smem);
        
        int full_capacity = max_blocks_per_sm * prop.multiProcessorCount;
        
        if (prob_smem > 0) {
            // 问题数据在 shared memory → 无 L2 cache 压力，打满 SM
            pop_size = full_capacity;
        } else {
            // 问题数据在 global memory → 根据 L2 cache 容量估算合理并发度
            //
            // 模型：pop = L2_size / working_set_bytes
            //   所有 block 访问同一份只读数据，L2/ws 反映 cache 能支撑的并发度
            //
            // SM 下限策略：L2/ws >= sm_min/2 时拉升到 sm_min（允许一定 cache 压力换取种群多样性）
            //   ch150: L2/ws=70, sm_min=128 → 70 >= 64 → 拉升到 128 ✓（多样性优先）
            //   pcb442: L2/ws=8, sm_min=128 → 8 < 64 → 不拉升 ✓（避免 thrashing）
            
            size_t ws = prob.working_set_bytes();
            if (ws > 0) {
                int l2_pop = (int)((size_t)prop.l2CacheSize / ws);
                pop_size = (l2_pop < full_capacity) ? l2_pop : full_capacity;
            } else {
                pop_size = full_capacity / 4;
            }
            
            int sm_min = 1;
            while (sm_min < prop.multiProcessorCount) sm_min *= 2;
            if (pop_size < sm_min) {
                bool l2_can_afford = (ws == 0) ||
                    ((size_t)prop.l2CacheSize / ws >= (size_t)sm_min / 2);
                if (l2_can_afford) pop_size = sm_min;
            }
        }
        
        // 向下取整到 2 的幂（warp 对齐、归约友好、islands 整除）
        {
            int p = 1;
            while (p * 2 <= pop_size) p *= 2;
            pop_size = p;
        }
        
        // 绝对下限：32（保证至少 1 岛 × 32 解的最小可用规模）
        if (pop_size < 32) pop_size = 32;
    }
    
    // 自适应岛屿数量（num_islands=0 时启用）
    int num_islands = cfg.num_islands;
    if (num_islands == 0) {
        // 策略：每岛至少 32 个个体，最多 8 岛
        // pop < 64   → 1 岛（纯 HC）
        // 64-127     → 2 岛
        // 128-255    → 4 岛
        // 256-511    → 8 岛
        // >= 512     → 8 岛
        if (pop_size < 64) {
            num_islands = 1;
        } else if (pop_size < 128) {
            num_islands = 2;
        } else if (pop_size < 256) {
            num_islands = 4;
        } else {
            num_islands = 8;
        }
    }
    
    bool use_islands = num_islands > 1;
    int island_size = use_islands ? pop_size / num_islands : pop_size;
    
    if (cfg.verbose) {
        const char* enc_name = pcfg.encoding == EncodingType::Permutation ? "Perm" 
                             : pcfg.encoding == EncodingType::Binary ? "Bin" : "Int";
        const char* strat_name = 
            cfg.migrate_strategy == MigrateStrategy::Ring ? "Ring" :
            cfg.migrate_strategy == MigrateStrategy::TopN ? "TopN" : "Hybrid";
        printf("\n[GenSolver v2.0 Block] %s%s [%d][%d] pop=%d%s gen=%d blk=%d",
               enc_name, pcfg.row_mode == RowMode::Partition ? "/Part" : "",
               pcfg.dim1, pcfg.row_mode == RowMode::Partition ? pcfg.total_elements : pcfg.dim2_default,
               pop_size, auto_pop ? "(auto)" : "",
               cfg.max_gen, block_threads);
        if (auto_pop) {
            size_t ws = prob.working_set_bytes();
            if (prob_smem > 0) {
                printf("\n  [AUTO] GPU=%s SM=%d strategy=full(smem) → pop=%d",
                       prop.name, prop.multiProcessorCount, pop_size);
            } else {
                printf("\n  [AUTO] GPU=%s SM=%d L2=%dKB ws=%zuKB → pop=%d",
                       prop.name, prop.multiProcessorCount,
                       prop.l2CacheSize / 1024, ws / 1024, pop_size);
            }
        }
        if (smem_overflow) {
            printf("\n  [WARN] Shared memory overflow, problem data stays in global memory");
        }
        if (use_islands) {
            if (cfg.num_islands == 0) {
                printf(" isl=%dx%d/%s(auto)", num_islands, island_size, strat_name);
            } else {
                printf(" isl=%dx%d/%s", num_islands, island_size, strat_name);
            }
        }
        if (use_sa)      printf(" SA=%.0f/%.4f", cfg.sa_temp_init, cfg.sa_alpha);
        if (use_crossover) printf(" CX=%.0f%%", cfg.crossover_rate * 100.0f);
        if (use_aos) printf(" AOS");
        if (use_time_limit) printf(" T=%.1fs", cfg.time_limit_sec);
        if (use_stagnation) printf(" stag=%d", cfg.stagnation_limit);
        if (num_init_solutions > 0) printf(" init=%d", num_init_solutions);
        if (cfg.use_cuda_graph) printf(" GRAPH");
        printf(" seed=%u\n", cfg.seed);
    }
    
    // --- 1. 分配 ---
    // crossover 栈需求（thread 0 在 local memory 中构造 child）
    if (use_crossover) {
        size_t ox_arrays = Sol::DIM1 * Sol::DIM2 * sizeof(bool)
                         + 512 * sizeof(bool)
                         + 512 * sizeof(int);
        size_t need = sizeof(Sol) + ox_arrays + 512;
        if (need > 1024) cudaDeviceSetLimit(cudaLimitStackSize, need);
    }
    
    ObjConfig oc = make_obj_config(pcfg);
    
    // --- 1b. 采样择优初始化 ---
    int oversample = cfg.init_oversample;
    if (oversample < 1) oversample = 1;
    int candidate_size = pop_size * oversample;
    bool do_oversample = (oversample > 1);
    
    Population<Sol> pop;
    
    if (do_oversample) {
        // 生成 K × pop_size 个候选解
        Population<Sol> candidates;
        candidates.allocate(candidate_size, block_threads);
        candidates.init_rng(cfg.seed, 256);
        candidates.init_population(pcfg, 256);
        
        // 启发式初始解注入（替换候选池尾部）
        if (pcfg.encoding == EncodingType::Permutation) {
            HeuristicMatrix heur_mats[8];
            int num_mats = prob.heuristic_matrices(heur_mats, 8);
            if (num_mats > 0) {
                bool is_partition = (pcfg.row_mode == RowMode::Partition);
                auto heur_sols = heuristic_init::build_from_matrices<Sol>(
                    heur_mats, num_mats, pcfg.dim1, pcfg.dim2_default, pcfg.encoding,
                    is_partition, pcfg.total_elements);
                int inject = (int)heur_sols.size();
                if (inject > candidate_size / 8) inject = candidate_size / 8;
                if (inject > 0) {
                    CUDA_CHECK(cudaMemcpy(
                        candidates.d_solutions + candidate_size - inject,
                        heur_sols.data(), sizeof(Sol) * inject,
                        cudaMemcpyHostToDevice));
                    if (cfg.verbose) {
                        printf("  [INIT] injected %d heuristic solutions into candidate pool\n", inject);
                    }
                }
            }
        }
        
        // GPU 上评估所有候选
        {
            size_t eval_smem = prob.shared_mem_bytes();
            if (eval_smem > 48 * 1024) {
                cudaFuncSetAttribute(evaluate_kernel<Problem, Sol>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)eval_smem);
            }
            int eval_grid = calc_grid_size(candidate_size, block_threads);
            evaluate_kernel<<<eval_grid, block_threads, eval_smem>>>(
                prob, candidates.d_solutions, candidate_size, eval_smem);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // 下载所有候选解到 host
        Sol* h_candidates = new Sol[candidate_size];
        CUDA_CHECK(cudaMemcpy(h_candidates, candidates.d_solutions,
                              sizeof(Sol) * candidate_size, cudaMemcpyDeviceToHost));
        
        // 构建候选信息
        std::vector<init_sel::CandidateInfo> cand_info(candidate_size);
        for (int i = 0; i < candidate_size; i++) {
            cand_info[i].idx = i;
            cand_info[i].penalty = h_candidates[i].penalty;
            cand_info[i].rank = 0;
            cand_info[i].crowding = 0.0f;
            cand_info[i].selected = false;
            for (int m = 0; m < oc.num_obj; m++) {
                cand_info[i].objs[m] = normalize_obj(
                    h_candidates[i].objectives[m], oc.dirs[m]);
            }
        }
        
        // 计算目标重要性
        float importance[MAX_OBJ];
        compute_importance(oc, importance);
        
        // 纯随机保底名额
        int num_random = (int)(pop_size * cfg.init_random_ratio);
        if (num_random < 1) num_random = 1;
        if (num_random > pop_size / 2) num_random = pop_size / 2;
        
        // 选择
        std::vector<int> selected;
        if (oc.num_obj == 1) {
            selected = init_sel::top_n_select(cand_info, pop_size, num_random);
        } else {
            selected = init_sel::nsga2_select(cand_info, oc.num_obj, importance,
                                               pop_size, num_random);
        }
        
        // 分配最终种群
        pop.allocate(pop_size, block_threads);
        // 复用候选的 RNG 状态（取前 pop_size 份）
        // 重新初始化 RNG 更安全（候选的 RNG 状态已被使用过）
        pop.init_rng(cfg.seed + 1, 256);
        
        // 上传选中的解到种群前部
        int num_selected = (int)selected.size();
        for (int i = 0; i < num_selected; i++) {
            CUDA_CHECK(cudaMemcpy(pop.d_solutions + i,
                                  candidates.d_solutions + selected[i],
                                  sizeof(Sol), cudaMemcpyDeviceToDevice));
        }
        
        // 剩余位置（纯随机保底）：从候选中随机选未被选中的
        // 简单做法：直接用候选中排在后面的未选中解
        if (num_selected < pop_size) {
            int fill_idx = num_selected;
            for (int i = 0; i < candidate_size && fill_idx < pop_size; i++) {
                if (!cand_info[i].selected) {
                    CUDA_CHECK(cudaMemcpy(pop.d_solutions + fill_idx,
                                          candidates.d_solutions + i,
                                          sizeof(Sol), cudaMemcpyDeviceToDevice));
                    fill_idx++;
                }
            }
        }
        
        if (cfg.verbose) {
            // 统计选中解的平均质量 vs 全部候选的平均质量
            float sel_avg = 0.0f, all_avg = 0.0f;
            for (int i = 0; i < candidate_size; i++) all_avg += cand_info[i].objs[0];
            all_avg /= candidate_size;
            for (int i = 0; i < num_selected; i++) sel_avg += cand_info[selected[i]].objs[0];
            if (num_selected > 0) sel_avg /= num_selected;
            
            const char* method = (oc.num_obj > 1) ? "NSGA-II" : "top-N";
            printf("  [INIT] oversample=%dx → %d candidates, %s select %d + %d random",
                   oversample, candidate_size, method, num_selected,
                   pop_size - num_selected);
            printf(" (obj0 avg: %.1f → %.1f, %.1f%% better)\n",
                   all_avg, sel_avg,
                   all_avg != 0.0f ? (1.0f - sel_avg / all_avg) * 100.0f : 0.0f);
        }
        
        delete[] h_candidates;
        // candidates 析构自动释放 GPU 内存
    } else {
        // oversample=1：纯随机，和之前一样
        pop.allocate(pop_size, block_threads);
        pop.init_rng(cfg.seed, 256);
        pop.init_population(pcfg, 256);
    }
    
    // --- 1c. 注入用户提供的初始解 ---
    // 策略：校验合法性 → 合法解替换种群尾部（保留 oversample 选出的好解在前部）
    if (init_solutions && num_init_solutions > 0) {
        int max_inject = pop_size / 16;  // 最多占种群 ~6%（保留多样性）
        if (max_inject < 1) max_inject = 1;
        if (max_inject > 16) max_inject = 16;  // 绝对上限
        int want = num_init_solutions;
        if (want > max_inject) want = max_inject;
        
        int injected = 0;
        for (int i = 0; i < want; i++) {
            const Sol& s = init_solutions[i];
            bool valid = true;
            
            // 基本维度检查
            for (int r = 0; r < pcfg.dim1 && valid; r++) {
                if (s.dim2_sizes[r] < 0 || s.dim2_sizes[r] > Sol::DIM2) {
                    valid = false; break;
                }
            }
            
            // 编码特定检查
            if (valid && pcfg.encoding == EncodingType::Permutation) {
                if (pcfg.row_mode == RowMode::Partition) {
                    // 分区模式：跨行元素不重复，总数 = total_elements
                    bool seen[512] = {};
                    int total = 0;
                    for (int r = 0; r < pcfg.dim1 && valid; r++) {
                        for (int c = 0; c < s.dim2_sizes[r] && valid; c++) {
                            int v = s.data[r][c];
                            if (v < 0 || v >= pcfg.total_elements) { valid = false; break; }
                            if (v < 512 && seen[v]) { valid = false; break; }
                            if (v < 512) seen[v] = true;
                            total++;
                        }
                    }
                    if (valid && total != pcfg.total_elements) valid = false;
                } else if (pcfg.perm_repeat_count > 1) {
                    // 多重集排列：每行中每个值 [0, N) 恰好出现 repeat_count 次
                    int R = pcfg.perm_repeat_count;
                    int N = pcfg.dim2_default / R;
                    for (int r = 0; r < pcfg.dim1 && valid; r++) {
                        if (s.dim2_sizes[r] != pcfg.dim2_default) { valid = false; break; }
                        int cnt[512] = {};
                        for (int c = 0; c < s.dim2_sizes[r] && valid; c++) {
                            int v = s.data[r][c];
                            if (v < 0 || v >= N) { valid = false; break; }
                            if (v < 512) cnt[v]++;
                        }
                        if (valid) {
                            for (int v = 0; v < N && v < 512 && valid; v++)
                                if (cnt[v] != R) valid = false;
                        }
                    }
                } else {
                    // 标准排列：每行元素 [0, dim2_default) 不重复
                    for (int r = 0; r < pcfg.dim1 && valid; r++) {
                        if (s.dim2_sizes[r] != pcfg.dim2_default) { valid = false; break; }
                        bool seen[512] = {};
                        for (int c = 0; c < s.dim2_sizes[r] && valid; c++) {
                            int v = s.data[r][c];
                            if (v < 0 || v >= pcfg.dim2_default) { valid = false; break; }
                            if (v < 512 && seen[v]) { valid = false; break; }
                            if (v < 512) seen[v] = true;
                        }
                    }
                }
            } else if (valid && pcfg.encoding == EncodingType::Binary) {
                for (int r = 0; r < pcfg.dim1 && valid; r++) {
                    for (int c = 0; c < s.dim2_sizes[r] && valid; c++) {
                        if (s.data[r][c] != 0 && s.data[r][c] != 1) { valid = false; break; }
                    }
                }
            }
            
            if (valid) {
                // 注入到种群尾部（从后往前填，保留前部的 oversample 好解）
                int target_idx = pop_size - 1 - injected;
                CUDA_CHECK(cudaMemcpy(pop.d_solutions + target_idx, &s,
                                      sizeof(Sol), cudaMemcpyHostToDevice));
                injected++;
            } else if (cfg.verbose) {
                printf("  [INIT] user solution #%d invalid, skipped\n", i);
            }
        }
        if (cfg.verbose && injected > 0) {
            printf("  [INIT] injected %d/%d user solutions (tail of population)\n",
                   injected, num_init_solutions);
        }
    }
    
    // v3.0: 构建序列注册表（替代旧的 d_op_weights）
    ProblemProfile profile = classify_problem(pcfg);
    SeqRegistry seq_reg = build_seq_registry(profile);

    if (custom_registry_fn) {
        custom_registry_fn(seq_reg);
    }
    
    // v3.1: K 步配置（多步执行）
    KStepConfig kstep = build_kstep_config();
    
    if (cfg.verbose) {
        const char* scale_names[] = {"Small", "Medium", "Large"};
        const char* struct_names[] = {"SingleSeq", "MultiFixed", "MultiPartition"};
        printf("  [PROFILE] scale=%s structure=%s\n",
               scale_names[(int)profile.scale], struct_names[(int)profile.structure]);
        printf("  [SEQ] %d sequences registered:", seq_reg.count);
        for (int i = 0; i < seq_reg.count; i++)
            printf(" %d(%.2f)", seq_reg.ids[i], seq_reg.weights[i]);
        printf("\n");
        printf("  [K-STEP] K weights: K1=%.2f K2=%.2f K3=%.2f\n",
               kstep.weights[0], kstep.weights[1], kstep.weights[2]);
    }
    
    int* d_best_idx;
    CUDA_CHECK(cudaMalloc(&d_best_idx, sizeof(int)));
    
    Sol* d_global_best = nullptr;
    if (use_sa) {
        CUDA_CHECK(cudaMalloc(&d_global_best, sizeof(Sol)));
        // v5.0 方案 B3: 导出 d_global_best 指针供外部读取（可选）
        if (d_global_best_out != nullptr) {
            *d_global_best_out = d_global_best;
        }
    }
    
    // AOS: 分配全局内存统计缓冲区（序列级粒度）
    AOSStats* d_aos_stats = nullptr;
    AOSStats* h_aos_stats = nullptr;
    
    if (use_aos) {
        CUDA_CHECK(cudaMalloc(&d_aos_stats, sizeof(AOSStats) * pop_size));
        h_aos_stats = new AOSStats[pop_size];
    }
    
    // --- 关系矩阵（G/O）：用于 SEQ_LNS_GUIDED_REBUILD ---
    // 仅 Permutation 编码 + 有 GUIDED_REBUILD 序列时启用
    bool use_relation_matrix = false;
    RelationMatrix rel_mat = {};
    int rel_N = 0;
    if (pcfg.encoding == EncodingType::Permutation) {
        for (int i = 0; i < seq_reg.count; i++) {
            if (seq_reg.ids[i] == seq::SEQ_LNS_GUIDED_REBUILD) {
                use_relation_matrix = true;
                break;
            }
        }
    }
    if (use_relation_matrix) {
        // N = dim2_default（排列中的元素数）
        rel_N = pcfg.dim2_default;
        if (rel_N > 0) {
            rel_mat = relation_matrix_create(rel_N, 0.95f);
            // 让用户提供先验知识初始化 G/O（可选，默认不做任何事）
            prob.init_relation_matrix(rel_mat.h_G, rel_mat.h_O, rel_N);
            relation_matrix_upload(rel_mat);
        } else {
            use_relation_matrix = false;
        }
    }
    
    // grid = pop_size（每个 block 处理一个解）
    int grid = pop_size;
    
    // --- 2. 初始评估 ---
    // 采样择优路径中已经评估过候选，但最终种群可能包含随机解，需要重新评估
    {
        size_t eval_smem = prob.shared_mem_bytes();
        if (eval_smem > 48 * 1024) {
            cudaFuncSetAttribute(evaluate_kernel<Problem, Sol>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)eval_smem);
        }
        int eval_grid = calc_grid_size(pop_size, block_threads);
        evaluate_kernel<<<eval_grid, block_threads, eval_smem>>>(
            prob, pop.d_solutions, pop_size, eval_smem);
    }
    
    if (use_sa) {
        find_best_kernel<<<1, 1>>>(pop.d_solutions, pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_global_best, pop.d_solutions + idx, sizeof(Sol), cudaMemcpyDeviceToDevice));
    }
    
    // --- 3. 主循环 ---
    // batch 大小决定了 AOS/关系矩阵/收敛检测的更新频率
    // 需要平衡：太小 → 同步开销大，太大 → 反应迟钝
    int batch;
    if (use_islands)
        batch = cfg.migrate_interval;
    else if (cfg.verbose)
        batch = cfg.print_every;
    else
        batch = cfg.max_gen;
    
    // 需要定期更新的功能：强制 batch ≤ 200
    if (use_relation_matrix || use_aos || use_time_limit || use_stagnation) {
        if (batch > 200) batch = 200;
    }
    
    int gen_done = 0;
    int migrate_round = 0;
    StopReason stop_reason = StopReason::MaxGen;
    
    // 收敛检测状态
    float prev_best_scalar = 1e30f;
    int stagnation_count = 0;
    
    // --- EvolveParams: 可变参数（device memory）---
    EvolveParams h_params;
    h_params.temp_start = 0.0f;
    h_params.gens_per_batch = batch;
    h_params.seq_reg = seq_reg;
    h_params.kstep = kstep;
    h_params.migrate_round = 0;
    h_params.oc = oc;
    
    EvolveParams* d_params = nullptr;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(EvolveParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &h_params, sizeof(EvolveParams), cudaMemcpyHostToDevice));
    
    // --- CUDA Graph ---
    const bool use_graph = cfg.use_cuda_graph;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStream_t stream = nullptr;
    
    if (use_graph) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    // lambda: 在 stream 上发射一个 batch 的 GPU kernel 序列
    auto launch_batch_kernels = [&](cudaStream_t s) {
        evolve_block_kernel<<<grid, block_threads, total_smem, s>>>(
            prob, pop.d_solutions, pop_size,
            pcfg.encoding, pcfg.dim1,
            oc, pop.d_rng_states,
            cfg.sa_alpha, prob_smem,
            d_aos_stats,
            use_relation_matrix ? rel_mat.d_G : nullptr,
            use_relation_matrix ? rel_mat.d_O : nullptr,
            rel_N,
            pcfg.value_lower_bound, pcfg.value_upper_bound,
            d_params);
        
        if (use_crossover) {
            crossover_block_kernel<<<grid, block_threads, total_smem, s>>>(
                prob, pop.d_solutions, pop_size,
                pcfg.encoding, pcfg.dim1,
                oc, pop.d_rng_states,
                cfg.crossover_rate, prob_smem,
                pcfg.row_mode == RowMode::Partition ? pcfg.total_elements : pcfg.dim2_default);
        }
        
        if (use_islands) {
            migrate_kernel<<<1, 1, 0, s>>>(pop.d_solutions, pop_size,
                                            island_size, oc,
                                            cfg.migrate_strategy, d_params);
        }
        
        if (use_sa) {
            elite_inject_kernel<<<1, 1, 0, s>>>(pop.d_solutions, pop_size,
                                                  d_global_best, oc);
        }
    };
    
    // 捕获 CUDA Graph（首次）
    if (use_graph) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch_batch_kernels(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        // 5-arg form: compatible with CUDA 10+; 3-arg form requires CUDA 12+
#if CUDART_VERSION >= 12000
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));
#else
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
#endif
        if (cfg.verbose) printf("  [CUDA Graph] captured and instantiated\n");
    }
    
    cudaEvent_t t_start, t_stop;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_stop));
    CUDA_CHECK(cudaEventRecord(t_start));
    
    // 时间感知 AOS：窗口累积器
    int win_seq_usage[MAX_SEQ] = {};
    int win_seq_improve[MAX_SEQ] = {};
    int win_k_usage[MAX_K] = {};
    int win_k_improve[MAX_K] = {};
    int batch_count = 0;
    const int aos_interval = (cfg.aos_update_interval > 0) ? cfg.aos_update_interval : 1;
    
    // v4.0: 约束导向 + 分层搜索
    const bool use_constraint_directed = cfg.use_constraint_directed && use_aos;
    const bool use_phased_search = cfg.use_phased_search && use_aos;
    float base_max_w[MAX_SEQ];
    for (int i = 0; i < seq_reg.count; i++) base_max_w[i] = seq_reg.max_w[i];
    
    if (cfg.verbose && (use_constraint_directed || use_phased_search)) {
        printf("  [P2] constraint_directed=%s phased_search=%s\n",
               use_constraint_directed ? "ON" : "OFF",
               use_phased_search ? "ON" : "OFF");
        if (use_phased_search)
            printf("  [P2] phases: explore=[0,%.0f%%) transition=[%.0f%%,%.0f%%) refine=[%.0f%%,100%%]\n",
                   cfg.phase_explore_end * 100, cfg.phase_explore_end * 100,
                   cfg.phase_refine_start * 100, cfg.phase_refine_start * 100);
    }
    
    while (gen_done < cfg.max_gen) {
        int gens = batch;
        if (gen_done + gens > cfg.max_gen) gens = cfg.max_gen - gen_done;
        
        float temp = use_sa ? cfg.sa_temp_init * powf(cfg.sa_alpha, (float)gen_done) : 0.0f;
        
        // 更新 device 端可变参数
        h_params.temp_start = temp;
        h_params.gens_per_batch = gens;
        h_params.seq_reg = seq_reg;
        h_params.kstep = kstep;
        h_params.migrate_round = migrate_round;
        CUDA_CHECK(cudaMemcpy(d_params, &h_params, sizeof(EvolveParams), cudaMemcpyHostToDevice));
        
        // 发射 GPU kernel 序列
        if (use_graph) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            launch_batch_kernels(nullptr);
        }
        
        // v5.0 方案 B3: 被动注入检查（在 Graph 之外单独调用）
        // 注意：必须在 Graph 之外，因为 inject_buf 内容是动态变化的
        if (inject_buf != nullptr && use_islands) {
            inject_check_kernel<<<1, 1>>>(pop.d_solutions, pop_size,
                                           island_size, inject_buf, oc);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        gen_done += gens;
        if (use_islands) migrate_round++;
        batch_count++;
        
        // AOS: 两层权重更新（EMA）+ 停滞检测
        if (use_aos && (batch_count % aos_interval == 0)) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_aos_stats, d_aos_stats,
                                  sizeof(AOSStats) * pop_size,
                                  cudaMemcpyDeviceToHost));
            
            // --- 聚合当前 batch 的统计到窗口累积器 ---
            for (int b = 0; b < pop_size; b++) {
                for (int i = 0; i < seq_reg.count; i++) {
                    win_seq_usage[i] += h_aos_stats[b].usage[i];
                    win_seq_improve[i] += h_aos_stats[b].improvement[i];
                }
                for (int i = 0; i < MAX_K; i++) {
                    win_k_usage[i] += h_aos_stats[b].k_usage[i];
                    win_k_improve[i] += h_aos_stats[b].k_improvement[i];
                }
            }
            
            constexpr float AOS_ALPHA = 0.6f;
            
            // --- v4.0: 约束导向 — 计算种群约束违反率 ---
            float penalty_ratio = 0.0f;
            if (use_constraint_directed) {
                Sol* h_pop_snap = new Sol[pop_size];
                CUDA_CHECK(cudaMemcpy(h_pop_snap, pop.d_solutions,
                                      sizeof(Sol) * pop_size, cudaMemcpyDeviceToHost));
                int infeasible = 0;
                for (int b = 0; b < pop_size; b++) {
                    if (h_pop_snap[b].penalty > 0.0f) infeasible++;
                }
                penalty_ratio = (float)infeasible / (float)pop_size;
                delete[] h_pop_snap;
            }
            
            // --- v4.0: 分层搜索 — 计算当前阶段的 floor/cap 调整 ---
            float phase_floor_mult = 1.0f;
            float phase_cap_mult   = 1.0f;
            if (use_phased_search) {
                float progress;
                if (use_time_limit && cfg.time_limit_sec > 0.0f) {
                    float elapsed_ms = 0.0f;
                    CUDA_CHECK(cudaEventRecord(t_stop));
                    CUDA_CHECK(cudaEventSynchronize(t_stop));
                    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));
                    progress = elapsed_ms / (cfg.time_limit_sec * 1000.0f);
                    if (progress > 1.0f) progress = 1.0f;
                } else {
                    progress = (float)gen_done / (float)cfg.max_gen;
                }
                if (progress < cfg.phase_explore_end) {
                    phase_floor_mult = 1.5f;   // 探索期：抬高 floor → 更均匀
                    phase_cap_mult   = 0.7f;   // 探索期：压低 cap → 防止过早集中
                } else if (progress >= cfg.phase_refine_start) {
                    phase_floor_mult = 0.5f;   // 精细期：降低 floor → 允许弱算子退出
                    phase_cap_mult   = 1.5f;   // 精细期：抬高 cap → 集中利用强算子
                }
            }
            
            // --- 第二层：算子权重更新（EMA） ---
            {
                float new_w[MAX_SEQ];
                // 延迟归一化：EMA 更新 + 边界约束（不归一化）
                for (int i = 0; i < seq_reg.count; i++) {
                    float signal = (win_seq_usage[i] > 0)
                        ? (float)win_seq_improve[i] / (float)win_seq_usage[i]
                        : 0.0f;
                    new_w[i] = AOS_ALPHA * seq_reg.weights[i]
                             + (1.0f - AOS_ALPHA) * (signal + AOS_WEIGHT_FLOOR);
                }
                
                float uniform = 1.0f / seq_reg.count;
                float base_floor = cfg.aos_weight_floor / seq_reg.count;
                if (base_floor < uniform * 0.5f) base_floor = uniform * 0.5f;
                float floor_val = base_floor * phase_floor_mult;
                float global_cap = cfg.aos_weight_cap * phase_cap_mult;
                
                // --- v4.0: 约束导向 — boost 跨行/行级算子权重 + 放宽 cap ---
                if (use_constraint_directed && penalty_ratio > 0.1f) {
                    float boost = 1.0f + (penalty_ratio - 0.1f) / 0.9f
                                  * (cfg.constraint_boost_max - 1.0f);
                    for (int i = 0; i < seq_reg.count; i++) {
                        if (seq_reg.categories[i] == SeqCategory::CrossRow ||
                            seq_reg.categories[i] == SeqCategory::RowLevel) {
                            new_w[i] *= boost;
                            float orig = (base_max_w[i] > 0.0f) ? base_max_w[i] : AOS_WEIGHT_CAP;
                            seq_reg.max_w[i] = orig * boost;
                        }
                    }
                } else if (use_constraint_directed) {
                    for (int i = 0; i < seq_reg.count; i++)
                        seq_reg.max_w[i] = base_max_w[i];
                }
                
                // 应用边界约束（不归一化）
                float sum = 0.0f;
                for (int i = 0; i < seq_reg.count; i++) {
                    float cap_val = (seq_reg.max_w[i] > 0.0f) ? seq_reg.max_w[i] : global_cap;
                    seq_reg.weights[i] = fmaxf(floor_val, fminf(cap_val, new_w[i]));
                    sum += seq_reg.weights[i];
                }
                
                // 更新缓存的权重和
                seq_reg.weights_sum = sum;
            }
            
            // --- 第一层：K 步数权重更新（EMA + 延迟归一化） ---
            {
                float new_w[MAX_K];
                for (int i = 0; i < MAX_K; i++) {
                    float rate = (win_k_usage[i] > 0)
                        ? (float)win_k_improve[i] / (float)win_k_usage[i]
                        : 0.0f;
                    new_w[i] = AOS_ALPHA * kstep.weights[i]
                             + (1.0f - AOS_ALPHA) * (rate + AOS_WEIGHT_FLOOR);
                }
                
                // 应用边界约束（不归一化）
                float floor_val = cfg.aos_weight_floor;
                float cap_val = 0.95f;
                for (int i = 0; i < MAX_K; i++) {
                    kstep.weights[i] = fmaxf(floor_val, fminf(cap_val, new_w[i]));
                }
                
                // K 步权重归一化（保持原有行为，因为 K 步选择不使用轮盘赌）
                float sum = 0.0f;
                for (int i = 0; i < MAX_K; i++) sum += kstep.weights[i];
                if (sum > 0.0f) {
                    for (int i = 0; i < MAX_K; i++)
                        kstep.weights[i] /= sum;
                }
            }
            
            // --- Debug: 前 5 个 batch 打印统计 ---
            if (cfg.verbose && gen_done <= batch * 5) {
                fprintf(stderr, "  [AOS batch g=%d] usage:", gen_done);
                for (int i = 0; i < seq_reg.count; i++) fprintf(stderr, " %d", win_seq_usage[i]);
                fprintf(stderr, " | improve:");
                for (int i = 0; i < seq_reg.count; i++) fprintf(stderr, " %d", win_seq_improve[i]);
                fprintf(stderr, " | w:");
                for (int i = 0; i < seq_reg.count; i++) fprintf(stderr, " %.3f", seq_reg.weights[i]);
                fprintf(stderr, " | sum=%.3f", seq_reg.weights_sum);
                fprintf(stderr, " | K: %.2f/%.2f/%.2f stag=%d",
                        kstep.weights[0], kstep.weights[1], kstep.weights[2], kstep.stagnation_count);
                if (use_constraint_directed)
                    fprintf(stderr, " | pen=%.1f%%", penalty_ratio * 100.0f);
                if (use_phased_search)
                    fprintf(stderr, " | phase_f=%.2f phase_c=%.2f", phase_floor_mult, phase_cap_mult);
                fprintf(stderr, "\n");
            }
            
            
            // --- 停滞检测 ---
            {
                int total_improve_all = 0;
                for (int i = 0; i < seq_reg.count; i++)
                    total_improve_all += win_seq_improve[i];
                
                if (total_improve_all == 0) {
                    kstep.stagnation_count++;
                } else {
                    kstep.stagnation_count = 0;
                }
                
                if (kstep.stagnation_count >= kstep.stagnation_limit) {
                    kstep.weights[0] = 0.80f;
                    kstep.weights[1] = 0.15f;
                    kstep.weights[2] = 0.05f;
                    kstep.stagnation_count = 0;
                }
            }
            
            // --- 清零窗口累积器 ---
            memset(win_seq_usage, 0, sizeof(win_seq_usage));
            memset(win_seq_improve, 0, sizeof(win_seq_improve));
            memset(win_k_usage, 0, sizeof(win_k_usage));
            memset(win_k_improve, 0, sizeof(win_k_improve));
        }
        
        // --- 关系矩阵更新（每个 batch 间隙，从种群 top-K 解统计）---
        // 多个好解贡献 G/O 信号，加速矩阵信息积累
        if (use_relation_matrix) {
            if (!use_aos) {
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            
            // 下载整个种群的目标值，找 top-K
            constexpr int REL_TOP_K = 4;
            int top_indices[REL_TOP_K];
            {
                // 简单方法：下载所有解的 scalar 目标，host 端排序取 top-K
                float* h_scores = new float[pop_size];
                Sol* h_pop_ptr = new Sol[pop_size];
                CUDA_CHECK(cudaMemcpy(h_pop_ptr, pop.d_solutions,
                                      sizeof(Sol) * pop_size, cudaMemcpyDeviceToHost));
                for (int b = 0; b < pop_size; b++) {
                    h_scores[b] = scalar_objective(h_pop_ptr[b], oc);
                    if (h_pop_ptr[b].penalty > 0.0f) h_scores[b] = 1e30f;
                }
                // 找 top-K 最小值
                for (int k = 0; k < REL_TOP_K && k < pop_size; k++) {
                    int mi = 0;
                    for (int b = 1; b < pop_size; b++) {
                        if (h_scores[b] < h_scores[mi]) mi = b;
                    }
                    top_indices[k] = mi;
                    h_scores[mi] = 1e30f;  // 标记已选
                }
                // 从 top-K 解更新 G/O
                int actual_k = (pop_size < REL_TOP_K) ? pop_size : REL_TOP_K;
                for (int k = 0; k < actual_k; k++) {
                    relation_matrix_update(rel_mat, h_pop_ptr[top_indices[k]], pcfg.dim1);
                }
                delete[] h_scores;
                delete[] h_pop_ptr;
            }
            
            relation_matrix_upload(rel_mat);
        }
        
        // 交叉 / 迁移 / 精英注入 已在 launch_batch_kernels 中统一发射
        
        // --- 时间限制检查 ---
        if (use_time_limit) {
            CUDA_CHECK(cudaEventRecord(t_stop));
            CUDA_CHECK(cudaEventSynchronize(t_stop));
            float ms_so_far = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms_so_far, t_start, t_stop));
            if (ms_so_far >= cfg.time_limit_sec * 1000.0f) {
                stop_reason = StopReason::TimeLimit;
                if (cfg.verbose) printf("  [STOP] time limit %.1fs reached at gen %d\n",
                                         cfg.time_limit_sec, gen_done);
                break;
            }
        }
        
        // --- 收敛检测 + reheat ---
        if (use_stagnation) {
            find_best_kernel<<<1, 1>>>(pop.d_solutions, pop_size, oc, d_best_idx);
            CUDA_CHECK(cudaDeviceSynchronize());
            int bi; CUDA_CHECK(cudaMemcpy(&bi, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
            Sol cur_best = pop.download_solution(bi);
            float cur_scalar = scalar_objective(cur_best, oc);
            if (cur_best.penalty > 0.0f) cur_scalar = 1e30f;
            
            constexpr float IMPROVE_EPS = 1e-6f;
            if (prev_best_scalar - cur_scalar > IMPROVE_EPS) {
                prev_best_scalar = cur_scalar;
                stagnation_count = 0;
            } else {
                stagnation_count++;
            }
            
            if (stagnation_count >= cfg.stagnation_limit) {
                if (use_sa && cfg.reheat_ratio > 0.0f) {
                    // reheat：将温度恢复到初始温度的 reheat_ratio 倍
                    // 通过回退 gen_done 实现（温度 = init * alpha^gen_done）
                    float target_temp = cfg.sa_temp_init * cfg.reheat_ratio;
                    int reheat_gen = (int)(logf(target_temp / cfg.sa_temp_init) / logf(cfg.sa_alpha));
                    if (reheat_gen < 0) reheat_gen = 0;
                    // 不真正回退 gen_done（会影响终止条件），而是记录一个 temp_offset
                    // 简化做法：直接在下一轮 batch 中 temp 会自然从 reheat 后的值开始
                    // 这里通过修改 gen_done 的等效温度来实现
                    if (cfg.verbose) {
                        float cur_temp = cfg.sa_temp_init * powf(cfg.sa_alpha, (float)gen_done);
                        printf("  [REHEAT] stagnation=%d at gen %d, temp %.4f → %.4f\n",
                               cfg.stagnation_limit, gen_done, cur_temp, target_temp);
                    }
                    // 将 gen_done 回退到对应 target_temp 的位置（但不超过已完成代数的一半）
                    int min_gen = gen_done / 2;
                    if (reheat_gen < min_gen) reheat_gen = min_gen;
                    gen_done = reheat_gen;
                    stagnation_count = 0;
                } else {
                    // 无 SA 时，收敛检测触发 → 提前终止
                    stop_reason = StopReason::Stagnation;
                    if (cfg.verbose) printf("  [STOP] stagnation=%d at gen %d, no SA to reheat\n",
                                             cfg.stagnation_limit, gen_done);
                    break;
                }
            }
        }
        
        // 打印进度
        if (cfg.verbose && gen_done % cfg.print_every == 0) {
            if (!use_stagnation) {
                find_best_kernel<<<1, 1>>>(pop.d_solutions, pop_size, oc, d_best_idx);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
            Sol best = pop.download_solution(idx);
            printf("  [%5d]", gen_done);
            for (int i = 0; i < pcfg.num_objectives; i++)
                printf(" %.1f", best.objectives[i]);
            if (best.penalty > 0.0f) printf(" P=%.1f", best.penalty);
            printf("\n");
        }
    }
    
    CUDA_CHECK(cudaEventRecord(t_stop));
    CUDA_CHECK(cudaEventSynchronize(t_stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));
    
    // --- 4. 最终结果 ---
    Sol best;
    if (use_sa) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&best, d_global_best, sizeof(Sol), cudaMemcpyDeviceToHost));
    } else {
        find_best_kernel<<<1, 1>>>(pop.d_solutions, pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int h_best_idx;
        CUDA_CHECK(cudaMemcpy(&h_best_idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        best = pop.download_solution(h_best_idx);
    }
    
    if (cfg.verbose) {
        const char* reason_str = stop_reason == StopReason::TimeLimit ? " [time]" :
                                 stop_reason == StopReason::Stagnation ? " [stag]" : "";
        printf("  Result:");
        for (int i = 0; i < pcfg.num_objectives; i++)
            printf(" obj%d=%.2f", i, best.objectives[i]);
        if (best.penalty > 0.0f) printf(" INFEASIBLE(%.2f)", best.penalty);
        printf("  %.0fms %dgen%s\n", elapsed_ms, gen_done, reason_str);
    }
    
    if (cfg.verbose) {
        for (int r = 0; r < pcfg.dim1; r++) {
            printf("  row[%d]:", r);
            int show = best.dim2_sizes[r] < 20 ? best.dim2_sizes[r] : 20;
            for (int c = 0; c < show; c++) printf(" %d", best.data[r][c]);
            if (best.dim2_sizes[r] > 20) printf(" ...(%d)", best.dim2_sizes[r]);
            printf("\n");
        }
    }
    
    // AOS: 打印最终两层权重
    if (use_aos && cfg.verbose) {
        printf("  AOS K-step weights: K1=%.3f K2=%.3f K3=%.3f\n",
               kstep.weights[0], kstep.weights[1], kstep.weights[2]);
        printf("  AOS seq weights:");
        for (int i = 0; i < seq_reg.count; i++)
            printf(" [%d]=%.3f", seq_reg.ids[i], seq_reg.weights[i]);
        printf("\n");
    }
    
    // 填充返回值
    result.best_solution = best;
    result.elapsed_ms = elapsed_ms;
    result.generations = gen_done;
    result.stop_reason = stop_reason;
    
    CUDA_CHECK(cudaFree(d_best_idx));
    if (d_global_best) CUDA_CHECK(cudaFree(d_global_best));
    if (d_aos_stats) CUDA_CHECK(cudaFree(d_aos_stats));
    if (h_aos_stats) delete[] h_aos_stats;
    if (use_relation_matrix) relation_matrix_destroy(rel_mat);
    CUDA_CHECK(cudaFree(d_params));
    if (graph_exec) CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(t_start));
    CUDA_CHECK(cudaEventDestroy(t_stop));
    
    return result;
}
