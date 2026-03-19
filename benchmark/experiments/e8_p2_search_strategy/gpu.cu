/**
 * E8: P2 约束导向 + 分层搜索策略 A/B 测试
 *
 * 对比四种配置：
 *   baseline:    仅 AOS（当前默认）
 *   constraint:  AOS + 约束导向
 *   phased:      AOS + 分层搜索
 *   combined:    AOS + 约束导向 + 分层搜索
 *
 * 测试问题：
 *   - VRP A-n32-k5（中等约束）
 *   - VRPTW 8客户（高约束：容量+时间窗）
 *   - Priority-VRP A-n32-k5（高约束：容量+优先级偏序）
 *   - TSP eil51（无约束 baseline，验证无回退）
 *
 * 时间预算：5s, 15s
 */
#include "bench_common.cuh"

struct PriorityVRPProblem : ProblemBase<PriorityVRPProblem, 8, 64> {
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
        size_t total = (size_t)stride * stride * sizeof(float)
                     + (size_t)n * sizeof(float) + (size_t)n * sizeof(int);
        return total <= SMEM_LIMIT ? total : 0;
    }
    size_t working_set_bytes() const {
        return (size_t)stride * stride * sizeof(float)
             + (size_t)n * sizeof(float) + (size_t)n * sizeof(int);
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
                                      const int* h_priority, int n, float cap,
                                      int nv, int mv) {
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

static const int an32k5_priority[AN32K5_N] = {
    2,2,2,2,2,2,2,2,2,2, 1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0
};

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

static void run_vrp() {
    fprintf(stderr, "\n=== VRP A-n32-k5 ===\n");
    float dist[AN32K5_NODES * AN32K5_NODES];
    compute_euc2d_dist(dist, an32k5_coords, AN32K5_NODES);

    float budgets[] = {5.0f, 15.0f};
    for (float t : budgets) {
        for (int v = 0; v < NUM_VARIANTS; v++) {
            char cfg_name[64];
            snprintf(cfg_name, sizeof(cfg_name), "%s_%.0fs", VARIANTS[v].name, t);
            SolverConfig c = make_p2_config(t, VARIANTS[v]);
            bench_run_recreate("VRP-A32k5", cfg_name,
                [&]() { return VRPProblem::create(dist, an32k5_demands, AN32K5_N, 100.0f, 5, 5); },
                c, 784.0f);
        }
    }
}

static void run_vrptw() {
    fprintf(stderr, "\n=== VRPTW 8-customer ===\n");

    const int N = 8;
    const int NODES = N + 1;
    float coords[NODES][2] = {
        {40,40}, {22,22},{36,26},{21,45},{45,35},{55,20},{33,34},{50,50},{55,45}
    };
    float demand[N] = {10,20,10,10,20,10,20,10};
    float earliest[NODES] = {0,  0,  5,  0, 10,  0,  0, 15,  0};
    float latest[NODES]   = {999,50,40,60,80,45,70,90,55};
    float service[NODES]  = {0, 10,10,10,10,10,10,10,10};
    float capacity = 40.0f;
    int num_vehicles = 3, max_vehicles = 3;

    float dist[NODES * NODES];
    for (int i = 0; i < NODES; i++)
        for (int j = 0; j < NODES; j++) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            dist[i * NODES + j] = sqrtf(dx*dx + dy*dy);
        }

    float budgets[] = {5.0f, 15.0f};
    for (float t : budgets) {
        for (int v = 0; v < NUM_VARIANTS; v++) {
            char cfg_name[64];
            snprintf(cfg_name, sizeof(cfg_name), "%s_%.0fs", VARIANTS[v].name, t);
            SolverConfig c = make_p2_config(t, VARIANTS[v]);
            bench_run_recreate("VRPTW-8", cfg_name,
                [&]() {
                    return VRPTWProblem::create(
                        dist, demand, earliest, latest, service,
                        N, capacity, num_vehicles, max_vehicles);
                },
                c, 0.0f);
        }
    }
}

static void run_priority_vrp() {
    fprintf(stderr, "\n=== Priority-VRP A-n32-k5 ===\n");
    float dist[AN32K5_NODES * AN32K5_NODES];
    compute_euc2d_dist(dist, an32k5_coords, AN32K5_NODES);

    float budgets[] = {5.0f, 15.0f};
    for (float t : budgets) {
        for (int v = 0; v < NUM_VARIANTS; v++) {
            char cfg_name[64];
            snprintf(cfg_name, sizeof(cfg_name), "%s_%.0fs", VARIANTS[v].name, t);
            SolverConfig c = make_p2_config(t, VARIANTS[v]);
            bench_run_recreate("PrioVRP-A32k5", cfg_name,
                [&]() {
                    return PriorityVRPProblem::create(
                        dist, an32k5_demands, an32k5_priority,
                        AN32K5_N, 100.0f, 5, 5);
                },
                c, 784.0f);
        }
    }
}

static void run_tsp_sanity() {
    fprintf(stderr, "\n=== TSP eil51 (sanity check, no constraints) ===\n");
    float dist[EIL51_N * EIL51_N];
    compute_euc2d_dist(dist, eil51_coords, EIL51_N);

    float budgets[] = {5.0f};
    for (float t : budgets) {
        for (int v = 0; v < NUM_VARIANTS; v++) {
            char cfg_name[64];
            snprintf(cfg_name, sizeof(cfg_name), "%s_%.0fs", VARIANTS[v].name, t);
            SolverConfig c = make_p2_config(t, VARIANTS[v]);
            bench_run_tsp<void>("eil51", cfg_name, EIL51_N, dist, c, 426.0f, 3);
        }
    }
}

int main() {
    bench_init();
    bench_csv_header();
    run_vrp();
    run_vrptw();
    run_priority_vrp();
    run_tsp_sanity();
    fprintf(stderr, "\n[e8] P2 search strategy A/B test completed.\n");
    return 0;
}
