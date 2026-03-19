/**
 * E2: GenSolver vs 专用求解器 (OR-Tools Routing) — GPU 侧
 *
 * 目的：参考对比，诚实展示与专用求解器的差距，强调通用性价值
 * 实例：TSP (全部 6 个 TSPLIB), VRP (A-n32-k5)
 * 时间预算：1s, 5s, 10s, 30s, 60s
 * 输出：CSV
 *
 * 用法：./gpu [tsp|vrp|all]
 */
#include "bench_common.cuh"

static void run_tsp() {
    float time_budgets[] = {1.0f, 5.0f, 10.0f, 30.0f, 60.0f};

    for (int i = 0; i < NUM_TSP_INSTANCES; i++) {
        auto& inst = ALL_TSP_INSTANCES[i];
        fprintf(stderr, "  [e2] TSP %s (n=%d)\n", inst.name, inst.n);

        float* dist = new float[inst.n * inst.n];
        compute_euc2d_dist(dist, inst.coords, inst.n);

        for (float t : time_budgets) {
            char cfg[64];
            snprintf(cfg, sizeof(cfg), "gensolver_%.0fs", t);
            SolverConfig c = make_timed_config(t);
            bench_run_tsp<void>(inst.name, cfg, inst.n, dist, c, inst.optimal);
        }
        delete[] dist;
    }
}

static void run_vrp() {
    fprintf(stderr, "  [e2] VRP A-n32-k5\n");
    float dist[AN32K5_NODES * AN32K5_NODES];
    compute_euc2d_dist(dist, an32k5_coords, AN32K5_NODES);

    float time_budgets[] = {1.0f, 5.0f, 10.0f, 30.0f};
    for (float t : time_budgets) {
        char cfg[64];
        snprintf(cfg, sizeof(cfg), "gensolver_%.0fs", t);
        SolverConfig c = make_timed_config(t);
        bench_run_recreate("A-n32-k5", cfg,
            [&]() { return VRPProblem::create(dist, an32k5_demands, AN32K5_N, 100.0f, 5, 5); },
            c, 784.0f);
    }
}

int main(int argc, char** argv) {
    const char* target = (argc > 1) ? argv[1] : "all";
    bench_init();
    bench_csv_header();

    bool all = (strcmp(target, "all") == 0);
    if (all || strcmp(target, "tsp") == 0) run_tsp();
    if (all || strcmp(target, "vrp") == 0) run_vrp();

    fprintf(stderr, "\n[e2] GPU side completed.\n");
    return 0;
}
