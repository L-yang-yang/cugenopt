/**
 * E8v2: P2 约束导向 + 分层搜索 — 大规模 & 紧约束实验
 *
 * 设计思路：
 *   - 用更大实例 + 更短时间，确保搜索无法完全收敛
 *   - VRPTW-20: 20 客户 4 车，紧时间窗 + 容量约束
 *   - PrioVRP-50: 50 客户 8 车（随机坐标），优先级偏序约束
 *   - 时间预算：1s, 3s（短时间放大策略差异）
 *
 * 对比：baseline / constraint / phased / combined
 */
#include "bench_common.cuh"
#include <cstdlib>

// ============================================================
// PriorityVRPProblem（复用 e2.1 定义）
// ============================================================
struct PriorityVRPProblem : ProblemBase<PriorityVRPProblem, 16, 64> {
    const float* d_dist;
    const float* d_demand;
    const int*   d_priority;
    const float* h_dist;
    int n, stride;
    float capacity;
    int num_vehicles, max_vehicles;
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
    static constexpr ObjDef OBJ_DEFS[] = {{ObjDir::Minimize, 1.0f, 0.0f}};
    __device__ float compute_obj(int, const Sol& sol) const { return calc_total_distance(sol); }
    __device__ float compute_penalty(const Sol& sol) const {
        float pen = 0.0f;
        int active = 0;
        for (int r = 0; r < num_vehicles; r++) {
            int size = sol.dim2_sizes[r];
            if (size == 0) continue;
            active++;
            float load = 0.0f;
            for (int j = 0; j < size; j++) load += d_demand[sol.data[r][j]];
            if (load > capacity) pen += (load - capacity) * 100.0f;
            int min_prio_seen = 3;
            for (int j = 0; j < size; j++) {
                int p = d_priority[sol.data[r][j]];
                if (p > min_prio_seen) pen += (float)(p - min_prio_seen) * 50.0f;
                if (p < min_prio_seen) min_prio_seen = p;
            }
        }
        if (active > max_vehicles) pen += (float)(active - max_vehicles) * 1000.0f;
        return pen;
    }
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = num_vehicles; cfg.dim2_default = 0;
        fill_obj_config(cfg);
        cfg.cross_row_prob = 0.3f;
        cfg.row_mode = RowMode::Partition;
        cfg.total_elements = n;
        return cfg;
    }
    static constexpr size_t SMEM_LIMIT = 48 * 1024;
    size_t shared_mem_bytes() const {
        size_t total = (size_t)stride*stride*sizeof(float) + (size_t)n*sizeof(float) + (size_t)n*sizeof(int);
        return total <= SMEM_LIMIT ? total : 0;
    }
    size_t working_set_bytes() const {
        return (size_t)stride*stride*sizeof(float) + (size_t)n*sizeof(float) + (size_t)n*sizeof(int);
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
                float d = h_dist[(i+1)*stride+(j+1)];
                if (d > max_d) max_d = d;
            }
        if (max_d <= 0.0f) return;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                float d = h_dist[(i+1)*stride+(j+1)];
                float prox = 1.0f - d / max_d;
                G[i*N+j] = prox * 0.3f;
                O[i*N+j] = prox * 0.1f;
            }
    }
    static PriorityVRPProblem create(const float* h_dist_ptr, const float* h_demand,
                                      const int* h_priority, int n, float cap, int nv, int mv) {
        PriorityVRPProblem prob;
        prob.n = n; prob.stride = n+1; prob.capacity = cap;
        prob.num_vehicles = nv; prob.max_vehicles = mv;
        prob.cache = GpuCache::disabled(); prob.h_dist = h_dist_ptr;
        int nn = n+1;
        float* dd; CUDA_CHECK(cudaMalloc(&dd, sizeof(float)*nn*nn));
        CUDA_CHECK(cudaMemcpy(dd, h_dist_ptr, sizeof(float)*nn*nn, cudaMemcpyHostToDevice));
        prob.d_dist = dd;
        float* ddem; CUDA_CHECK(cudaMalloc(&ddem, sizeof(float)*n));
        CUDA_CHECK(cudaMemcpy(ddem, h_demand, sizeof(float)*n, cudaMemcpyHostToDevice));
        prob.d_demand = ddem;
        int* dpri; CUDA_CHECK(cudaMalloc(&dpri, sizeof(int)*n));
        CUDA_CHECK(cudaMemcpy(dpri, h_priority, sizeof(int)*n, cudaMemcpyHostToDevice));
        prob.d_priority = dpri;
        return prob;
    }
    void destroy() {
        if (d_dist)     { cudaFree(const_cast<float*>(d_dist));   d_dist = nullptr; }
        if (d_demand)   { cudaFree(const_cast<float*>(d_demand)); d_demand = nullptr; }
        if (d_priority) { cudaFree(const_cast<int*>(d_priority)); d_priority = nullptr; }
        h_dist = nullptr; cache.destroy();
    }
};

// ============================================================
// VRPTW-20: 20 客户 4 车，紧时间窗
// ============================================================
// 坐标在 [0,100]x[0,100] 区域，depot 在中心 (50,50)
// 时间窗故意设紧：窗口宽度 15-30，服务时间 5-10
// 容量 50，需求 5-15 → 平均每车 5 客户，容量紧张

static const int VRPTW20_N = 20;
static const int VRPTW20_NODES = 21;
static const float vrptw20_coords[VRPTW20_NODES][2] = {
    {50,50},  // depot
    {20,70},{35,80},{15,55},{40,65},{60,85},
    {75,70},{90,60},{80,45},{65,30},{50,20},
    {30,15},{15,30},{25,45},{45,40},{70,50},
    {85,75},{55,65},{35,35},{60,15},{80,25}
};
static const float vrptw20_demand[VRPTW20_N] = {
    8,12,7,10,15, 9,11,8,13,6, 10,14,7,12,9, 8,11,13,10,7
};
static const float vrptw20_earliest[VRPTW20_NODES] = {
    0,  5, 10,  0, 15, 20,  5, 25, 10,  0, 30,
    15,  0, 20, 10,  5, 25, 15,  0, 35, 20
};
static const float vrptw20_latest[VRPTW20_NODES] = {
    999, 25, 35, 20, 40, 50, 30, 55, 35, 25, 60,
     40, 25, 45, 35, 30, 55, 40, 25, 65, 45
};
static const float vrptw20_service[VRPTW20_NODES] = {
    0, 5,7,5,8,6, 7,5,8,6,5, 7,5,8,6,7, 5,8,6,7,5
};

// ============================================================
// 50 客户随机实例生成（确定性种子）
// ============================================================
static void gen_random_coords(float coords[][2], int n_nodes, unsigned seed) {
    srand(seed);
    coords[0][0] = 50.0f; coords[0][1] = 50.0f;
    for (int i = 1; i < n_nodes; i++) {
        coords[i][0] = (float)(rand() % 100);
        coords[i][1] = (float)(rand() % 100);
    }
}

static void gen_random_demand(float* demand, int n, unsigned seed) {
    srand(seed + 1000);
    for (int i = 0; i < n; i++)
        demand[i] = 5.0f + (float)(rand() % 11);  // [5, 15]
}

static void gen_random_priority(int* priority, int n, unsigned seed) {
    srand(seed + 2000);
    for (int i = 0; i < n; i++)
        priority[i] = rand() % 3;  // 0, 1, 2
}

// ============================================================
// 配置变体
// ============================================================
struct ConfigVariant {
    const char* name;
    bool constraint_directed;
    bool phased_search;
};

static const ConfigVariant VARIANTS[] = {
    {"baseline",   false, false},
    {"constraint", true,  false},
    {"phased",     false, true},
    {"combined",   true,  true},
};
static const int NUM_VARIANTS = 4;

static SolverConfig make_p2_config(float seconds, const ConfigVariant& v) {
    SolverConfig c = make_timed_config(seconds);
    c.use_constraint_directed = v.constraint_directed;
    c.use_phased_search       = v.phased_search;
    return c;
}

// ============================================================
// VRPTW-20 实验
// ============================================================
static void run_vrptw20() {
    fprintf(stderr, "\n=== VRPTW-20 (tight time windows) ===\n");

    float dist[VRPTW20_NODES * VRPTW20_NODES];
    for (int i = 0; i < VRPTW20_NODES; i++)
        for (int j = 0; j < VRPTW20_NODES; j++) {
            float dx = vrptw20_coords[i][0] - vrptw20_coords[j][0];
            float dy = vrptw20_coords[i][1] - vrptw20_coords[j][1];
            dist[i * VRPTW20_NODES + j] = sqrtf(dx*dx + dy*dy);
        }

    float budgets[] = {1.0f, 3.0f, 10.0f};
    for (float t : budgets) {
        for (int v = 0; v < NUM_VARIANTS; v++) {
            char cfg_name[64];
            snprintf(cfg_name, sizeof(cfg_name), "%s_%.0fs", VARIANTS[v].name, t);
            SolverConfig c = make_p2_config(t, VARIANTS[v]);
            bench_run_recreate("VRPTW-20", cfg_name,
                [&]() {
                    return VRPTWProblem::create(
                        dist, vrptw20_demand, vrptw20_earliest, vrptw20_latest,
                        vrptw20_service, VRPTW20_N, 50.0f, 4, 4);
                },
                c, 0.0f);
        }
    }
}

// ============================================================
// PrioVRP-50 实验
// ============================================================
static void run_prio_vrp50() {
    fprintf(stderr, "\n=== PrioVRP-50 (50 customers, priority constraints) ===\n");

    const int N = 50;
    const int NODES = N + 1;
    float coords[NODES][2];
    float demand[N];
    int priority[N];
    gen_random_coords(coords, NODES, 12345);
    gen_random_demand(demand, N, 12345);
    gen_random_priority(priority, N, 12345);

    float dist[NODES * NODES];
    for (int i = 0; i < NODES; i++)
        for (int j = 0; j < NODES; j++) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            dist[i * NODES + j] = sqrtf(dx*dx + dy*dy);
        }

    float budgets[] = {1.0f, 3.0f, 10.0f};
    for (float t : budgets) {
        for (int v = 0; v < NUM_VARIANTS; v++) {
            char cfg_name[64];
            snprintf(cfg_name, sizeof(cfg_name), "%s_%.0fs", VARIANTS[v].name, t);
            SolverConfig c = make_p2_config(t, VARIANTS[v]);
            bench_run_recreate("PrioVRP-50", cfg_name,
                [&]() {
                    return PriorityVRPProblem::create(
                        dist, demand, priority, N, 60.0f, 8, 10);
                },
                c, 0.0f);
        }
    }
}

// ============================================================
// VRP A-n32-k5 短时间（1s）— 验证短时间下是否有差异
// ============================================================
static void run_vrp_short() {
    fprintf(stderr, "\n=== VRP A-n32-k5 (short budget) ===\n");
    float dist[AN32K5_NODES * AN32K5_NODES];
    compute_euc2d_dist(dist, an32k5_coords, AN32K5_NODES);

    float budgets[] = {0.5f, 1.0f};
    for (float t : budgets) {
        for (int v = 0; v < NUM_VARIANTS; v++) {
            char cfg_name[64];
            snprintf(cfg_name, sizeof(cfg_name), "%s_%.1fs", VARIANTS[v].name, t);
            SolverConfig c = make_p2_config(t, VARIANTS[v]);
            bench_run_recreate("VRP-A32k5", cfg_name,
                [&]() { return VRPProblem::create(dist, an32k5_demands, AN32K5_N, 100.0f, 5, 5); },
                c, 784.0f);
        }
    }
}

int main() {
    bench_init();
    bench_csv_header();
    run_vrptw20();
    run_prio_vrp50();
    run_vrp_short();
    fprintf(stderr, "\n[e8v2] P2 search strategy large-scale test completed.\n");
    return 0;
}
