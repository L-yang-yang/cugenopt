/**
 * E7: 中等规模基准实验
 *
 * 目的：在中等规模标准基准实例上测试 cuGenOpt，为后续优化提供数据基线
 * 实例：
 *   - QAP:      nug12 (N=12, opt=578), tai15a (N=15, opt=388214)
 *   - JSP:      ft06 (6x6, opt=55), ft10 (10x10, opt=930)
 *   - Knapsack: knapPI_1_100 (N=100, cap=995)
 *   - VRPTW:    Solomon R101 (N=100, best=1637.7), C101 (N=100, best=827.3),
 *               RC101 (N=100, best=1619.8)
 * 配置：default (time_limit=30s)
 * 输出：CSV
 *
 * 用法：./gpu [data_dir]
 */
#include "bench_common.cuh"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

// ============================================================
// 文件解析工具
// ============================================================

struct QAPData {
    int n;
    std::vector<float> dist;
    std::vector<float> flow;
};

static QAPData parse_qaplib(const char* path) {
    QAPData d;
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    f >> d.n;
    int nn = d.n * d.n;
    d.dist.resize(nn);
    d.flow.resize(nn);
    for (int i = 0; i < nn; i++) f >> d.dist[i];
    for (int i = 0; i < nn; i++) f >> d.flow[i];
    return d;
}

struct JSPData {
    int num_jobs, num_machines;
    std::vector<int> machines;
    std::vector<float> durations;
};

static JSPData parse_jsp(const char* path) {
    JSPData d;
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    f >> d.num_jobs >> d.num_machines;
    int total = d.num_jobs * d.num_machines;
    d.machines.resize(total);
    d.durations.resize(total);
    for (int j = 0; j < d.num_jobs; j++) {
        for (int o = 0; o < d.num_machines; o++) {
            int m; float dur;
            f >> m >> dur;
            d.machines[j * d.num_machines + o] = m;
            d.durations[j * d.num_machines + o] = dur;
        }
    }
    return d;
}

struct KnapsackData {
    int n;
    float capacity;
    std::vector<float> values;
    std::vector<float> weights;
};

static KnapsackData parse_knapsack(const char* path) {
    KnapsackData d;
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int cap;
    f >> d.n >> cap;
    d.capacity = (float)cap;
    d.values.resize(d.n);
    d.weights.resize(d.n);
    for (int i = 0; i < d.n; i++) {
        int v, w;
        f >> v >> w;
        d.values[i] = (float)v;
        d.weights[i] = (float)w;
    }
    return d;
}

// ============================================================
// Solomon VRPTW 文件解析
// ============================================================

struct SolomonNode {
    int id;
    float x, y;
    float demand;
    float ready, due, service;
};

struct SolomonData {
    int num_vehicles;
    float capacity;
    std::vector<SolomonNode> nodes;  // nodes[0] = depot
    int num_customers;               // nodes.size() - 1
    std::vector<float> dist;         // (n+1)*(n+1) 距离矩阵
};

static SolomonData parse_solomon(const char* path) {
    SolomonData d;
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    std::string line;
    // skip instance name + blank
    std::getline(f, line);
    // skip until VEHICLE section
    while (std::getline(f, line)) {
        if (line.find("NUMBER") != std::string::npos && line.find("CAPACITY") != std::string::npos)
            break;
    }
    f >> d.num_vehicles >> d.capacity;
    // skip until CUSTOMER data
    while (std::getline(f, line)) {
        if (line.find("CUST") != std::string::npos) break;
    }
    std::getline(f, line); // skip blank line after header

    SolomonNode node;
    while (f >> node.id >> node.x >> node.y >> node.demand
             >> node.ready >> node.due >> node.service) {
        d.nodes.push_back(node);
    }

    d.num_customers = (int)d.nodes.size() - 1;
    int nn = (int)d.nodes.size();
    d.dist.resize(nn * nn);
    for (int i = 0; i < nn; i++)
        for (int j = 0; j < nn; j++) {
            float dx = d.nodes[i].x - d.nodes[j].x;
            float dy = d.nodes[i].y - d.nodes[j].y;
            d.dist[i * nn + j] = sqrtf(dx * dx + dy * dy);
        }
    return d;
}

// ============================================================
// VRPTW Problem (D1=25, D2=128, 支持 N<=100 客户, <=25 辆车)
// ============================================================

struct VRPTWMedium : ProblemBase<VRPTWMedium, 25, 128> {
    const float* d_dist;
    const float* d_demand;
    const float* d_earliest;
    const float* d_latest;
    const float* d_service;
    const float* h_dist;   // host-side distance matrix for heuristic init
    int n;          // 客户数（不含 depot）
    int stride;     // n+1
    float capacity;
    int num_vehicles;
    int max_vehicles;

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
    __device__ float compute_obj(int, const Sol& sol) const {
        return calc_total_distance(sol);
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

            float time = 0.0f;
            int prev = 0;
            for (int j = 0; j < size; j++) {
                int node = sol.data[r][j] + 1;
                float travel = d_dist[prev * stride + node];
                time += travel;
                if (time < d_earliest[node])
                    time = d_earliest[node];
                if (time > d_latest[node])
                    penalty += (time - d_latest[node]) * 50.0f;
                time += d_service[node];
                prev = node;
            }
            float return_time = time + d_dist[prev * stride + 0];
            if (return_time > d_latest[0])
                penalty += (return_time - d_latest[0]) * 50.0f;
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

    int heuristic_matrices(HeuristicMatrix* out, int max_count) const {
        if (max_count < 1 || !h_dist) return 0;
        out[0] = {h_dist, stride};
        return 1;
    }

    size_t shared_mem_bytes() const {
        size_t dist_bytes = (size_t)stride * stride * sizeof(float);
        size_t aux_bytes  = (size_t)(n + 1) * 4 * sizeof(float);
        return dist_bytes + aux_bytes;
    }

    size_t working_set_bytes() const {
        return (size_t)stride * stride * sizeof(float) + (size_t)(n + 1) * 4 * sizeof(float);
    }

    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sd = reinterpret_cast<float*>(smem);
        int dist_size = stride * stride;
        for (int i = tid; i < dist_size; i += bsz) sd[i] = d_dist[i];
        d_dist = sd;

        float* sdem = sd + dist_size;
        for (int i = tid; i < n; i += bsz) sdem[i] = d_demand[i];
        d_demand = sdem;

        float* se = sdem + n;
        int nn = n + 1;
        for (int i = tid; i < nn; i += bsz) se[i] = d_earliest[i];
        d_earliest = se;

        float* sl = se + nn;
        for (int i = tid; i < nn; i += bsz) sl[i] = d_latest[i];
        d_latest = sl;

        float* ss = sl + nn;
        for (int i = tid; i < nn; i += bsz) ss[i] = d_service[i];
        d_service = ss;
    }

    static VRPTWMedium create(const SolomonData& sd) {
        VRPTWMedium p;
        p.n = sd.num_customers;
        p.stride = sd.num_customers + 1;
        p.capacity = sd.capacity;
        p.num_vehicles = sd.num_vehicles;
        p.max_vehicles = sd.num_vehicles;
        p.h_dist = sd.dist.data();

        int nn = p.stride;
        float *dd, *ddem, *de, *dl, *ds;
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * nn * nn));
        CUDA_CHECK(cudaMemcpy(dd, sd.dist.data(), sizeof(float) * nn * nn, cudaMemcpyHostToDevice));
        p.d_dist = dd;

        std::vector<float> demand(p.n), earliest(nn), latest(nn), service(nn);
        for (int i = 0; i < p.n; i++)
            demand[i] = sd.nodes[i + 1].demand;
        for (int i = 0; i < nn; i++) {
            earliest[i] = sd.nodes[i].ready;
            latest[i]   = sd.nodes[i].due;
            service[i]   = sd.nodes[i].service;
        }

        CUDA_CHECK(cudaMalloc(&ddem, sizeof(float) * p.n));
        CUDA_CHECK(cudaMemcpy(ddem, demand.data(), sizeof(float) * p.n, cudaMemcpyHostToDevice));
        p.d_demand = ddem;

        CUDA_CHECK(cudaMalloc(&de, sizeof(float) * nn));
        CUDA_CHECK(cudaMemcpy(de, earliest.data(), sizeof(float) * nn, cudaMemcpyHostToDevice));
        p.d_earliest = de;

        CUDA_CHECK(cudaMalloc(&dl, sizeof(float) * nn));
        CUDA_CHECK(cudaMemcpy(dl, latest.data(), sizeof(float) * nn, cudaMemcpyHostToDevice));
        p.d_latest = dl;

        CUDA_CHECK(cudaMalloc(&ds, sizeof(float) * nn));
        CUDA_CHECK(cudaMemcpy(ds, service.data(), sizeof(float) * nn, cudaMemcpyHostToDevice));
        p.d_service = ds;

        return p;
    }

    void destroy() {
        if (d_dist)     { cudaFree(const_cast<float*>(d_dist));     d_dist = nullptr; }
        if (d_demand)   { cudaFree(const_cast<float*>(d_demand));   d_demand = nullptr; }
        if (d_earliest) { cudaFree(const_cast<float*>(d_earliest)); d_earliest = nullptr; }
        if (d_latest)   { cudaFree(const_cast<float*>(d_latest));   d_latest = nullptr; }
        if (d_service)  { cudaFree(const_cast<float*>(d_service));  d_service = nullptr; }
    }
};

// ============================================================
// QAP Problem (D2=16, 支持 N<=16)
// ============================================================

struct QAPMedium : ProblemBase<QAPMedium, 1, 16> {
    const float* d_flow;
    const float* d_dist;
    int n;

    __device__ float calc_cost(const Sol& s) const {
        float cost = 0.0f;
        int sz = s.dim2_sizes[0];
        for (int i = 0; i < sz; i++)
            for (int j = 0; j < sz; j++)
                cost += d_flow[i * n + j] * d_dist[s.data[0][i] * n + s.data[0][j]];
        return cost;
    }

    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int, const Sol& s) const { return calc_cost(s); }
    __device__ float compute_penalty(const Sol&) const { return 0.0f; }

    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = 1; cfg.dim2_default = n;
        fill_obj_config(cfg);
        return cfg;
    }

    size_t shared_mem_bytes() const { return 2 * (size_t)n * n * sizeof(float); }

    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sf = reinterpret_cast<float*>(smem);
        float* sd = sf + n * n;
        int total = n * n;
        for (int i = tid; i < total; i += bsz) { sf[i] = d_flow[i]; sd[i] = d_dist[i]; }
        d_flow = sf; d_dist = sd;
    }

    static QAPMedium create(const float* h_flow, const float* h_dist, int n) {
        QAPMedium p;
        p.n = n;
        float *df, *dd;
        CUDA_CHECK(cudaMalloc(&df, sizeof(float) * n * n));
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * n * n));
        CUDA_CHECK(cudaMemcpy(df, h_flow, sizeof(float) * n * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dd, h_dist, sizeof(float) * n * n, cudaMemcpyHostToDevice));
        p.d_flow = df; p.d_dist = dd;
        return p;
    }

    void destroy() {
        if (d_flow) cudaFree(const_cast<float*>(d_flow));
        if (d_dist) cudaFree(const_cast<float*>(d_dist));
        d_flow = nullptr; d_dist = nullptr;
    }
};

// ============================================================
// JSP Perm Problem (D2=128, 支持 J*O<=128, J/M<=16)
// ============================================================

struct JSPPermMedium : ProblemBase<JSPPermMedium, 1, 128> {
    const int*   d_machine;
    const float* d_duration;
    int num_jobs, num_ops, num_machines;

    __device__ float decode_and_makespan(const Sol& s) const {
        int total = num_jobs * num_ops;
        int size = s.dim2_sizes[0];
        if (size < total) return 1e9f;

        float job_avail[16] = {};
        float mach_avail[16] = {};
        int   job_next_op[16] = {};

        float makespan = 0.0f;
        for (int k = 0; k < total; k++) {
            int j = s.data[0][k];
            if (j < 0 || j >= num_jobs) return 1e9f;
            int op = job_next_op[j];
            if (op >= num_ops) continue;

            int flat = j * num_ops + op;
            int m = d_machine[flat];
            float dur = d_duration[flat];
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
    __device__ float compute_obj(int, const Sol& s) const { return decode_and_makespan(s); }
    __device__ float compute_penalty(const Sol&) const { return 0.0f; }

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

    static JSPPermMedium create(const int* h_machine, const float* h_duration,
                                 int nj, int no, int nm) {
        JSPPermMedium p;
        p.num_jobs = nj; p.num_ops = no; p.num_machines = nm;
        int total = nj * no;
        int* dm; float* dd;
        CUDA_CHECK(cudaMalloc(&dm, sizeof(int) * total));
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * total));
        CUDA_CHECK(cudaMemcpy(dm, h_machine, sizeof(int) * total, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dd, h_duration, sizeof(float) * total, cudaMemcpyHostToDevice));
        p.d_machine = dm; p.d_duration = dd;
        return p;
    }

    void destroy() {
        if (d_machine)  { cudaFree(const_cast<int*>(d_machine));     d_machine = nullptr; }
        if (d_duration) { cudaFree(const_cast<float*>(d_duration));  d_duration = nullptr; }
    }
};

// ============================================================
// Knapsack Problem (D2=128, 支持 N<=128)
// ============================================================

struct KnapsackMedium : ProblemBase<KnapsackMedium, 1, 128> {
    const float* d_weights;
    const float* d_values;
    float capacity;
    int n;

    __device__ float calc_total_value(const Sol& s) const {
        float tv = 0.0f;
        int size = s.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            if (s.data[0][i]) tv += d_values[i];
        return tv;
    }

    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Maximize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int, const Sol& s) const { return calc_total_value(s); }

    __device__ float compute_penalty(const Sol& s) const {
        float tw = 0.0f;
        int size = s.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            if (s.data[0][i]) tw += d_weights[i];
        float over = tw - capacity;
        return (over > 0.0f) ? over : 0.0f;
    }

    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Binary;
        cfg.dim1 = 1; cfg.dim2_default = n;
        fill_obj_config(cfg);
        return cfg;
    }

    size_t shared_mem_bytes() const { return 2 * (size_t)n * sizeof(float); }

    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sw = reinterpret_cast<float*>(smem);
        float* sv = sw + n;
        for (int i = tid; i < n; i += bsz) { sw[i] = d_weights[i]; sv[i] = d_values[i]; }
        d_weights = sw; d_values = sv;
    }

    static KnapsackMedium create(const float* hw, const float* hv, int n, float cap) {
        KnapsackMedium p;
        p.n = n; p.capacity = cap;
        float *dw, *dv;
        CUDA_CHECK(cudaMalloc(&dw, sizeof(float) * n));
        CUDA_CHECK(cudaMalloc(&dv, sizeof(float) * n));
        CUDA_CHECK(cudaMemcpy(dw, hw, sizeof(float) * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv, hv, sizeof(float) * n, cudaMemcpyHostToDevice));
        p.d_weights = dw; p.d_values = dv;
        return p;
    }

    void destroy() {
        if (d_weights) cudaFree(const_cast<float*>(d_weights));
        if (d_values)  cudaFree(const_cast<float*>(d_values));
        d_weights = nullptr; d_values = nullptr;
    }
};

// ============================================================
// Knapsack 最优解参考值（动态规划精确求解）
// ============================================================

static int knapsack_dp_optimal(const KnapsackData& d) {
    int cap = (int)d.capacity;
    std::vector<int> dp(cap + 1, 0);
    for (int i = 0; i < d.n; i++) {
        int w = (int)d.weights[i], v = (int)d.values[i];
        for (int c = cap; c >= w; c--)
            if (dp[c - w] + v > dp[c])
                dp[c] = dp[c - w] + v;
    }
    return dp[cap];
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    bench_init();
    bench_csv_header();

    const float TIME = 30.0f;
    const char* cfg_name = "default_t30s";

    const char* data_dir = "../../data";
    if (argc > 1) data_dir = argv[1];

    // --- QAP: nug12 ---
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/qaplib/nug12.dat", data_dir);
        QAPData d = parse_qaplib(path);
        fprintf(stderr, "[e7] QAP nug12: N=%d\n", d.n);
        auto p = QAPMedium::create(d.flow.data(), d.dist.data(), d.n);
        SolverConfig c = make_timed_config(TIME);
        bench_run("QAP_nug12", cfg_name, p, c, 578.0f);
        p.destroy();
    }

    // --- QAP: tai15a ---
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/qaplib/tai15a.dat", data_dir);
        QAPData d = parse_qaplib(path);
        fprintf(stderr, "[e7] QAP tai15a: N=%d\n", d.n);
        auto p = QAPMedium::create(d.flow.data(), d.dist.data(), d.n);
        SolverConfig c = make_timed_config(TIME);
        bench_run("QAP_tai15a", cfg_name, p, c, 388214.0f);
        p.destroy();
    }

    // --- JSP: ft06 (6x6, opt=55) ---
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/jsp/ft06.txt", data_dir);
        JSPData d = parse_jsp(path);
        fprintf(stderr, "[e7] JSP ft06: %dx%d\n", d.num_jobs, d.num_machines);
        auto p = JSPPermMedium::create(d.machines.data(), d.durations.data(),
                                        d.num_jobs, d.num_machines, d.num_machines);
        SolverConfig c = make_timed_config(TIME);
        bench_run("JSP_ft06_Perm", cfg_name, p, c, 55.0f);
        p.destroy();
    }

    // --- JSP: ft10 (10x10, opt=930) ---
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/jsp/ft10.txt", data_dir);
        JSPData d = parse_jsp(path);
        fprintf(stderr, "[e7] JSP ft10: %dx%d\n", d.num_jobs, d.num_machines);
        auto p = JSPPermMedium::create(d.machines.data(), d.durations.data(),
                                        d.num_jobs, d.num_machines, d.num_machines);
        SolverConfig c = make_timed_config(TIME);
        bench_run("JSP_ft10_Perm", cfg_name, p, c, 930.0f);
        p.destroy();
    }

    // --- Knapsack: knapPI_1_100 (N=100) ---
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/knapsack/knapPI_1_100.txt", data_dir);
        KnapsackData d = parse_knapsack(path);
        int opt = knapsack_dp_optimal(d);
        fprintf(stderr, "[e7] Knapsack N=%d, cap=%.0f, DP optimal=%d\n", d.n, d.capacity, opt);
        auto p = KnapsackMedium::create(d.weights.data(), d.values.data(), d.n, d.capacity);
        SolverConfig c = make_timed_config(TIME);
        bench_run("Knapsack100", cfg_name, p, c, -(float)opt);
        p.destroy();
    }

    // --- VRPTW: Solomon R101 (N=100, best known distance = 1637.7) ---
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/solomon/R101.txt", data_dir);
        SolomonData sd = parse_solomon(path);
        fprintf(stderr, "[e7] VRPTW R101: N=%d, vehicles=%d, cap=%.0f\n",
                sd.num_customers, sd.num_vehicles, sd.capacity);
        auto p = VRPTWMedium::create(sd);
        SolverConfig c = make_timed_config(TIME);
        bench_run("VRPTW_R101", cfg_name, p, c, 1637.7f);
        p.destroy();
    }

    // --- VRPTW: Solomon C101 (N=100, best known distance = 827.3) ---
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/solomon/C101.txt", data_dir);
        SolomonData sd = parse_solomon(path);
        fprintf(stderr, "[e7] VRPTW C101: N=%d, vehicles=%d, cap=%.0f\n",
                sd.num_customers, sd.num_vehicles, sd.capacity);
        auto p = VRPTWMedium::create(sd);
        SolverConfig c = make_timed_config(TIME);
        bench_run("VRPTW_C101", cfg_name, p, c, 827.3f);
        p.destroy();
    }

    // --- VRPTW: Solomon RC101 (N=100, best known distance = 1619.8) ---
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/solomon/RC101.txt", data_dir);
        SolomonData sd = parse_solomon(path);
        fprintf(stderr, "[e7] VRPTW RC101: N=%d, vehicles=%d, cap=%.0f\n",
                sd.num_customers, sd.num_vehicles, sd.capacity);
        auto p = VRPTWMedium::create(sd);
        SolverConfig c = make_timed_config(TIME);
        bench_run("VRPTW_RC101", cfg_name, p, c, 1619.8f);
        p.destroy();
    }

    fprintf(stderr, "\n[e7] Medium-scale benchmark completed.\n");
    return 0;
}
