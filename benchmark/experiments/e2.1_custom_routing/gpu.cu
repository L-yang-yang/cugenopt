/**
 * E2.1: 自定义路径规划 — OR-Tools Routing 无法支持的场景
 *
 * 场景 A：带优先级约束的 VRP (Priority-Constrained VRP)
 *   - 约束扩展：penalty 中加入优先级偏序约束
 *   - OR-Tools 的 Dimension 机制无法表达路径内偏序
 *
 * 场景 B：非线性运输成本 VRP (Nonlinear-Cost VRP)
 *   - 目标扩展：边成本随累积负载非线性增长 cost = dist * (1 + 0.3 * load_ratio²)
 *   - OR-Tools 的 ArcCostEvaluator 只接受 (from, to)，无法访问累积负载
 *
 * 实例：基于 A-n32-k5
 * 时间预算：1s, 10s, 60s
 * 输出：CSV (instance,config,seed,obj,penalty,time_ms,gap_pct,generations,stop_reason)
 */
#include "bench_common.cuh"

// ============================================================
// PriorityVRPProblem：在 VRPProblem 基础上增加优先级偏序约束
// ============================================================
struct PriorityVRPProblem : ProblemBase<PriorityVRPProblem, 8, 64> {
    const float* d_dist;
    const float* d_demand;
    const int*   d_priority;   // 0=low, 1=medium, 2=high
    const float* h_dist;
    int n;
    int stride;
    float capacity;
    int num_vehicles;
    int max_vehicles;
    GpuCache cache;

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

    __device__ float calc_total_distance(const Sol& sol) const {
        float total = 0.0f;
        for (int r = 0; r < num_vehicles; r++)
            total += compute_route_dist(sol.data[r], sol.dim2_sizes[r]);
        return total;
    }

    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        return calc_total_distance(sol);
    }

    __device__ float compute_penalty(const Sol& sol) const {
        float pen = 0.0f;
        int active = 0;
        for (int r = 0; r < num_vehicles; r++) {
            int size = sol.dim2_sizes[r];
            if (size == 0) continue;
            active++;

            // 容量约束
            float load = 0.0f;
            for (int j = 0; j < size; j++)
                load += d_demand[sol.data[r][j]];
            if (load > capacity)
                pen += (load - capacity) * 100.0f;

            // 优先级偏序约束：路径内高优先级必须在低优先级之前
            int min_prio_seen = 3;
            for (int j = 0; j < size; j++) {
                int p = d_priority[sol.data[r][j]];
                if (p > min_prio_seen) {
                    // 当前客户优先级高于前面已出现的最低优先级 → 违规
                    pen += (float)(p - min_prio_seen) * 50.0f;
                }
                if (p < min_prio_seen) min_prio_seen = p;
            }
        }
        if (active > max_vehicles)
            pen += (float)(active - max_vehicles) * 1000.0f;
        return pen;
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

    static constexpr size_t SMEM_LIMIT = 48 * 1024;
    size_t shared_mem_bytes() const {
        size_t total = (size_t)stride * stride * sizeof(float)
                     + (size_t)n * sizeof(float)
                     + (size_t)n * sizeof(int);
        return total <= SMEM_LIMIT ? total : 0;
    }
    size_t working_set_bytes() const {
        return (size_t)stride * stride * sizeof(float)
             + (size_t)n * sizeof(float)
             + (size_t)n * sizeof(int);
    }
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sd = reinterpret_cast<float*>(smem);
        int dist_size = stride * stride;
        for (int i = tid; i < dist_size; i += bsz) sd[i] = d_dist[i];
        d_dist = sd;
        float* sdem = sd + dist_size;
        for (int i = tid; i < n; i += bsz) sdem[i] = d_demand[i];
        d_demand = sdem;
        int* spri = reinterpret_cast<int*>(sdem + n);
        for (int i = tid; i < n; i += bsz) spri[i] = d_priority[i];
        d_priority = spri;
    }

    void init_relation_matrix(float* G, float* O, int N) const {
        if (!h_dist || N != n) return;
        float max_d = 0.0f;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float d = h_dist[(i + 1) * stride + (j + 1)];
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

    static PriorityVRPProblem create(const float* h_dist_ptr, const float* h_demand,
                                      const int* h_priority, int n, float capacity,
                                      int num_vehicles, int max_vehicles) {
        PriorityVRPProblem prob;
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

        int* dpri;
        CUDA_CHECK(cudaMalloc(&dpri, sizeof(int) * n));
        CUDA_CHECK(cudaMemcpy(dpri, h_priority, sizeof(int) * n, cudaMemcpyHostToDevice));
        prob.d_priority = dpri;

        return prob;
    }

    void destroy() {
        if (d_dist)     { cudaFree(const_cast<float*>(d_dist));   d_dist = nullptr; }
        if (d_demand)   { cudaFree(const_cast<float*>(d_demand)); d_demand = nullptr; }
        if (d_priority) { cudaFree(const_cast<int*>(d_priority)); d_priority = nullptr; }
        h_dist = nullptr;
        cache.destroy();
    }
};

// ============================================================
// NonlinearCostVRPProblem：边成本随累积负载非线性增长
//   cost(edge) = dist(i,j) * (1.0 + 0.3 * (load/capacity)²)
//   模拟真实场景：车辆越重，油耗/电耗越高
//   OR-Tools 的 ArcCostEvaluator 只接受 (from, to)，无法访问累积负载
// ============================================================
struct NonlinearCostVRPProblem : ProblemBase<NonlinearCostVRPProblem, 8, 64> {
    const float* d_dist;
    const float* d_demand;
    const float* h_dist;
    int n;
    int stride;
    float capacity;
    int num_vehicles;
    int max_vehicles;
    GpuCache cache;

    __device__ float compute_route_nonlinear_cost(const int* route, int size) const {
        if (size == 0) return 0.0f;
        float cost = 0.0f;
        float load = 0.0f;
        int prev = 0;
        for (int j = 0; j < size; j++) {
            int cust = route[j];
            int node = cust + 1;
            load += d_demand[cust];
            float ratio = load / capacity;
            float edge_dist = d_dist[prev * stride + node];
            cost += edge_dist * (1.0f + 0.3f * ratio * ratio);
            prev = node;
        }
        cost += d_dist[prev * stride + 0];  // 返回 depot（空载，系数 1.0）
        return cost;
    }

    __device__ float calc_total_cost(const Sol& sol) const {
        float total = 0.0f;
        for (int r = 0; r < num_vehicles; r++)
            total += compute_route_nonlinear_cost(sol.data[r], sol.dim2_sizes[r]);
        return total;
    }

    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        return calc_total_cost(sol);
    }

    __device__ float compute_penalty(const Sol& sol) const {
        float pen = 0.0f;
        int active = 0;
        for (int r = 0; r < num_vehicles; r++) {
            int size = sol.dim2_sizes[r];
            if (size == 0) continue;
            active++;
            float load = 0.0f;
            for (int j = 0; j < size; j++)
                load += d_demand[sol.data[r][j]];
            if (load > capacity)
                pen += (load - capacity) * 100.0f;
        }
        if (active > max_vehicles)
            pen += (float)(active - max_vehicles) * 1000.0f;
        return pen;
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

    static constexpr size_t SMEM_LIMIT = 48 * 1024;
    size_t shared_mem_bytes() const {
        size_t total = (size_t)stride * stride * sizeof(float)
                     + (size_t)n * sizeof(float);
        return total <= SMEM_LIMIT ? total : 0;
    }
    size_t working_set_bytes() const {
        return (size_t)stride * stride * sizeof(float)
             + (size_t)n * sizeof(float);
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

    void init_relation_matrix(float* G, float* O, int N) const {
        if (!h_dist || N != n) return;
        float max_d = 0.0f;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float d = h_dist[(i + 1) * stride + (j + 1)];
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

    static NonlinearCostVRPProblem create(const float* h_dist_ptr, const float* h_demand,
                                           int n, float capacity,
                                           int num_vehicles, int max_vehicles) {
        NonlinearCostVRPProblem prob;
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

// ============================================================
// A-n32-k5 优先级分配（确定性，可复现）
// 31 个客户分为 3 档：high(2)=10, medium(1)=11, low(0)=10
// 分配规则：客户 0-9 → high, 10-20 → medium, 21-30 → low
// ============================================================
static const int an32k5_priority[AN32K5_N] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   // customers 0-9: high
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // customers 10-20: medium
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0     // customers 21-30: low
};

static void run_priority_vrp() {
    fprintf(stderr, "  [e2.1] Priority-VRP A-n32-k5\n");
    float dist[AN32K5_NODES * AN32K5_NODES];
    compute_euc2d_dist(dist, an32k5_coords, AN32K5_NODES);

    float time_budgets[] = {1.0f, 10.0f, 60.0f};
    for (float t : time_budgets) {
        char cfg[64];
        snprintf(cfg, sizeof(cfg), "gensolver_pvrp_%.0fs", t);
        SolverConfig c = make_timed_config(t);
        bench_run_recreate("A-n32-k5-prio", cfg,
            [&]() {
                return PriorityVRPProblem::create(
                    dist, an32k5_demands, an32k5_priority,
                    AN32K5_N, 100.0f, 5, 5);
            }, c, 784.0f);
    }
}

// 同时跑标准 VRP 作为 baseline（无优先级约束时的最优距离）
static void run_standard_vrp() {
    fprintf(stderr, "  [e2.1] Standard-VRP A-n32-k5 (baseline)\n");
    float dist[AN32K5_NODES * AN32K5_NODES];
    compute_euc2d_dist(dist, an32k5_coords, AN32K5_NODES);

    float time_budgets[] = {1.0f, 10.0f, 60.0f};
    for (float t : time_budgets) {
        char cfg[64];
        snprintf(cfg, sizeof(cfg), "gensolver_vrp_%.0fs", t);
        SolverConfig c = make_timed_config(t);
        bench_run_recreate("A-n32-k5-std", cfg,
            [&]() {
                return VRPProblem::create(dist, an32k5_demands, AN32K5_N, 100.0f, 5, 5);
            }, c, 784.0f);
    }
}

static void run_nonlinear_cost_vrp() {
    fprintf(stderr, "  [e2.1] Nonlinear-Cost-VRP A-n32-k5\n");
    float dist[AN32K5_NODES * AN32K5_NODES];
    compute_euc2d_dist(dist, an32k5_coords, AN32K5_NODES);

    float time_budgets[] = {1.0f, 10.0f, 60.0f};
    for (float t : time_budgets) {
        char cfg[64];
        snprintf(cfg, sizeof(cfg), "gensolver_nlvrp_%.0fs", t);
        SolverConfig c = make_timed_config(t);
        bench_run_recreate("A-n32-k5-nlcost", cfg,
            [&]() {
                return NonlinearCostVRPProblem::create(
                    dist, an32k5_demands, AN32K5_N, 100.0f, 5, 5);
            }, c, 0.0f);  // 无已知最优，gap 列输出 0
    }
}

int main() {
    bench_init();
    bench_csv_header();
    run_standard_vrp();
    run_priority_vrp();
    run_nonlinear_cost_vrp();
    fprintf(stderr, "\n[e2.1] GPU side completed.\n");
    return 0;
}
