/**
 * population.cuh - 种群管理
 * 
 * v2.0: Block 级架构
 *   - RNG 数组大小 = pop_size * block_size（每个 block 内每个线程独立 RNG）
 *   - 初始化 kernel 保持 1-thread-per-solution（初始化只做一次，不需要并行）
 *   - find_best_kernel 保持单线程（种群规模不大）
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

// ============================================================
// Device 端 Kernel（模板化）
// ============================================================

template<typename Sol>
__global__ void init_permutation_kernel(Sol* pop, int pop_size, 
                                         int dim1, int dim2_default,
                                         curandState* rng_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pop_size) return;
    Sol& sol = pop[tid];
    curandState* rng = &rng_states[tid];
    for (int r = 0; r < dim1; r++) {
        sol.dim2_sizes[r] = dim2_default;
        for (int c = 0; c < dim2_default; c++) sol.data[r][c] = c;
        shuffle(sol.data[r], dim2_default, rng);
    }
    sol.penalty = 0.0f;
}

template<typename Sol>
__global__ void init_binary_kernel(Sol* pop, int pop_size,
                                    int dim1, int dim2_default,
                                    curandState* rng_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pop_size) return;
    Sol& sol = pop[tid];
    curandState* rng = &rng_states[tid];
    for (int r = 0; r < dim1; r++) {
        sol.dim2_sizes[r] = dim2_default;
        for (int c = 0; c < dim2_default; c++) sol.data[r][c] = curand(rng) % 2;
    }
    sol.penalty = 0.0f;
}

template<typename Sol>
__global__ void init_integer_kernel(Sol* pop, int pop_size,
                                     int dim1, int dim2_default,
                                     int lb, int ub,
                                     curandState* rng_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pop_size) return;
    Sol& sol = pop[tid];
    curandState* rng = &rng_states[tid];
    int range = ub - lb + 1;
    for (int r = 0; r < dim1; r++) {
        sol.dim2_sizes[r] = dim2_default;
        for (int c = 0; c < dim2_default; c++)
            sol.data[r][c] = lb + (curand(rng) % range);
    }
    sol.penalty = 0.0f;
}

// ============================================================
// 多重集排列初始化 — 每个值 [0, N) 重复 R 次，总长度 N*R
// ============================================================
// 用于 JSP 工序排列编码：N=num_jobs, R=num_ops，值 j 出现 R 次表示工件 j

template<typename Sol>
__global__ void init_multiset_perm_kernel(Sol* pop, int pop_size,
                                           int dim1, int num_values, int repeat_count,
                                           curandState* rng_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pop_size) return;
    Sol& sol = pop[tid];
    curandState* rng = &rng_states[tid];
    int total = num_values * repeat_count;
    for (int r = 0; r < dim1; r++) {
        sol.dim2_sizes[r] = total;
        int idx = 0;
        for (int v = 0; v < num_values; v++)
            for (int k = 0; k < repeat_count; k++)
                sol.data[r][idx++] = v;
        shuffle(sol.data[r], total, rng);
    }
    sol.penalty = 0.0f;
}

// ============================================================
// 分区初始化 — 元素 {0..total_elements-1} 不重复分配到 dim1 行
// ============================================================

template<typename Sol>
__global__ void init_partition_kernel(Sol* pop, int pop_size,
                                      int dim1, int total_elements,
                                      curandState* rng_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pop_size) return;
    Sol& sol = pop[tid];
    curandState* rng = &rng_states[tid];
    
    for (int i = 0; i < total_elements; i++) sol.data[0][i] = i;
    shuffle(sol.data[0], total_elements, rng);
    
    int idx = 0;
    for (int r = 0; r < dim1; r++) {
        int count = total_elements / dim1;
        if (r < total_elements % dim1) count++;
        sol.dim2_sizes[r] = count;
        if (r > 0) {
            for (int c = 0; c < count; c++)
                sol.data[r][c] = sol.data[0][idx + c];
        }
        idx += count;
    }
    
    sol.penalty = 0.0f;
}

template<typename Sol>
__global__ void find_best_kernel(const Sol* pop, int pop_size,
                                  ObjConfig oc, int* best_idx) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int best = 0;
    for (int i = 1; i < pop_size; i++)
        if (is_better(pop[i], pop[best], oc)) best = i;
    *best_idx = best;
}

// ============================================================
// Host 端 RAII 类（模板化）
// ============================================================

template<typename Sol>
class Population {
public:
    Sol*         d_solutions  = nullptr;
    curandState* d_rng_states = nullptr;  // 大小 = pop_size * block_size
    int          size         = 0;
    int          rng_count    = 0;        // RNG 状态总数

    Population() = default;
    
    // block_size: Block 级架构下每个 block 的线程数
    // RNG 数组大小 = pop_size * block_size（每个 block 内每个线程独立 RNG）
    void allocate(int pop_size, int block_size = 128) {
        size = pop_size;
        rng_count = pop_size * block_size;
        CUDA_CHECK(cudaMalloc(&d_solutions, sizeof(Sol) * size));
        CUDA_CHECK(cudaMalloc(&d_rng_states, sizeof(curandState) * rng_count));
    }
    
    void init_rng(unsigned seed, int block_size = 256) {
        int grid = calc_grid_size(rng_count, block_size);
        init_curand_kernel<<<grid, block_size>>>(d_rng_states, seed, rng_count);
        CUDA_CHECK_LAST();
    }
    
    void init_population(const ProblemConfig& cfg, int block_size = 256) {
        int grid = calc_grid_size(size, block_size);
        
        if (cfg.row_mode == RowMode::Partition) {
            init_partition_kernel<<<grid, block_size>>>(
                d_solutions, size, cfg.dim1, cfg.total_elements, d_rng_states);
        } else if (cfg.encoding == EncodingType::Permutation && cfg.perm_repeat_count > 1) {
            int num_values = cfg.dim2_default / cfg.perm_repeat_count;
            init_multiset_perm_kernel<<<grid, block_size>>>(
                d_solutions, size, cfg.dim1, num_values, cfg.perm_repeat_count, d_rng_states);
        } else {
            switch (cfg.encoding) {
                case EncodingType::Permutation:
                    init_permutation_kernel<<<grid, block_size>>>(
                        d_solutions, size, cfg.dim1, cfg.dim2_default, d_rng_states);
                    break;
                case EncodingType::Binary:
                    init_binary_kernel<<<grid, block_size>>>(
                        d_solutions, size, cfg.dim1, cfg.dim2_default, d_rng_states);
                    break;
                case EncodingType::Integer:
                    init_integer_kernel<<<grid, block_size>>>(
                        d_solutions, size, cfg.dim1, cfg.dim2_default,
                        cfg.value_lower_bound, cfg.value_upper_bound,
                        d_rng_states);
                    break;
            }
        }
        CUDA_CHECK_LAST();
    }
    
    Sol download_solution(int idx) const {
        Sol h_sol;
        CUDA_CHECK(cudaMemcpy(&h_sol, d_solutions + idx, sizeof(Sol), cudaMemcpyDeviceToHost));
        return h_sol;
    }
    
    ~Population() {
        if (d_solutions)  cudaFree(d_solutions);
        if (d_rng_states) cudaFree(d_rng_states);
    }
    
    Population(const Population&) = delete;
    Population& operator=(const Population&) = delete;
    Population(Population&& o) noexcept 
        : d_solutions(o.d_solutions), d_rng_states(o.d_rng_states),
          size(o.size), rng_count(o.rng_count) {
        o.d_solutions = nullptr; o.d_rng_states = nullptr;
        o.size = 0; o.rng_count = 0;
    }
};
