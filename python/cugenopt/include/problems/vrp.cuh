/**
 * vrp.cuh - 容量约束车辆路径问题 (CVRP)
 * 
 * 继承 ProblemBase，使用 ObjDef 目标注册机制
 * 多行编码（D1=K 条路线，分区初始化 + 跨行算子）
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"
#include "gpu_cache.cuh"

struct VRPProblem : ProblemBase<VRPProblem, 8, 64> {
    // GPU 数据
    const float* d_dist;
    const float* d_demand;
    const float* h_dist;  // host 端距离矩阵（含 depot，用于 init_relation_matrix）
    int n;
    int stride;
    float capacity;
    int num_vehicles;
    int max_vehicles;
    GpuCache cache;
    
    // ---- 目标计算 ----
    __device__ float compute_route_dist(const int* route, int size) const {
        if (size == 0) return 0.0f;
        float dist = 0.0f;
        int prev = 0;
        for (int j = 0; j < size; j++) {
            int node = route[j] + 1;
            dist += d_dist[prev * stride + node];
            prev = node;
        }
        dist += d_dist[prev * stride + 0];
        return dist;
    }
    
    __device__ float eval_route(const int* route, int size) const {
        if (size == 0) return 0.0f;
        if (!cache.keys) return compute_route_dist(route, size);
        
        uint64_t key = route_hash(route, size);
        float dist;
        if (cache_lookup(cache, key, dist)) {
            atomicAdd(cache.d_hits, 1);
            return dist;
        }
        dist = compute_route_dist(route, size);
        cache_insert(cache, key, dist);
        atomicAdd(cache.d_misses, 1);
        return dist;
    }
    
    __device__ float calc_total_distance(const Sol& sol) const {
        float total = 0.0f;
        for (int r = 0; r < num_vehicles; r++)
            total += eval_route(sol.data[r], sol.dim2_sizes[r]);
        return total;
    }
    
    // ---- 目标定义（OBJ_DEFS 与 compute_obj 必须一一对应）----
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},   // case 0: calc_total_distance
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_total_distance(sol);   // OBJ_DEFS[0]
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        float penalty = 0.0f;
        int active = 0;
        for (int r = 0; r < num_vehicles; r++) {
            int size = sol.dim2_sizes[r];
            if (size == 0) continue;
            active++;
            float load = 0.0f;
            for (int j = 0; j < size; j++)
                load += d_demand[sol.data[r][j]];
            if (load > capacity)
                penalty += (load - capacity) * 100.0f;
        }
        if (active > max_vehicles)
            penalty += (float)(active - max_vehicles) * 1000.0f;
        return penalty;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = num_vehicles;
        cfg.dim2_default = 0;
        fill_obj_config(cfg);
        cfg.cross_row_prob = 0.3f;
        cfg.row_mode = RowMode::Partition;
        cfg.total_elements = n;
        return cfg;
    }
    
    // ---- shared memory 接口 ----
    static constexpr size_t SMEM_LIMIT = 48 * 1024;
    
    size_t shared_mem_bytes() const {
        size_t dist_bytes = (size_t)stride * stride * sizeof(float);
        size_t demand_bytes = (size_t)n * sizeof(float);
        size_t total = dist_bytes + demand_bytes;
        return total <= SMEM_LIMIT ? total : 0;
    }
    
    size_t working_set_bytes() const {
        return (size_t)stride * stride * sizeof(float) + (size_t)n * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sd = reinterpret_cast<float*>(smem);
        int dist_size = stride * stride;
        for (int i = tid; i < dist_size; i += bsz) sd[i] = d_dist[i];
        d_dist = sd;
        float* sdem = sd + dist_size;
        for (int i = tid; i < n; i += bsz) sdem[i] = d_demand[i];
        d_demand = sdem;
    }
    
    void enable_cache(int cap = 65536) { cache = GpuCache::allocate(cap); }
    void print_cache_stats() const { cache.print_stats(); }
    
    // 距离先验：客户间距离近 → G/O 分数高
    // 注意：h_dist 含 depot（stride×stride），元素编号 0..n-1 对应 node 1..n
    void init_relation_matrix(float* G, float* O, int N) const {
        if (!h_dist || N != n) return;
        float max_d = 0.0f;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float d = h_dist[(i + 1) * stride + (j + 1)];  // 跳过 depot
                if (d > max_d) max_d = d;
            }
        if (max_d <= 0.0f) return;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                float d = h_dist[(i + 1) * stride + (j + 1)];
                float proximity = 1.0f - d / max_d;
                G[i * N + j] = proximity * 0.3f;
                O[i * N + j] = proximity * 0.1f;
            }
    }
    
    static VRPProblem create(const float* h_dist_ptr, const float* h_demand,
                              int n, float capacity,
                              int num_vehicles, int max_vehicles) {
        VRPProblem prob;
        prob.n = n;
        prob.stride = n + 1;
        prob.capacity = capacity;
        prob.num_vehicles = num_vehicles;
        prob.max_vehicles = max_vehicles;
        prob.cache = GpuCache::disabled();
        prob.h_dist = h_dist_ptr;
        
        int n_nodes = n + 1;
        float* dd;
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * n_nodes * n_nodes));
        CUDA_CHECK(cudaMemcpy(dd, h_dist_ptr, sizeof(float) * n_nodes * n_nodes, cudaMemcpyHostToDevice));
        prob.d_dist = dd;
        
        float* ddem;
        CUDA_CHECK(cudaMalloc(&ddem, sizeof(float) * n));
        CUDA_CHECK(cudaMemcpy(ddem, h_demand, sizeof(float) * n, cudaMemcpyHostToDevice));
        prob.d_demand = ddem;
        
        return prob;
    }
    
    void destroy() {
        if (d_dist)   { cudaFree(const_cast<float*>(d_dist));   d_dist = nullptr; }
        if (d_demand) { cudaFree(const_cast<float*>(d_demand)); d_demand = nullptr; }
        h_dist = nullptr;
        cache.destroy();
    }
};
