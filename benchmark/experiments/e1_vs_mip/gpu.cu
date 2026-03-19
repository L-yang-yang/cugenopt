/**
 * E1: GenSolver vs 通用 MIP (SCIP/CBC) — GPU 侧
 *
 * 目的：证明在复杂约束问题上，GenSolver 比 MIP 更快找到可行解
 * 实例：TSP (N=51,100,150), VRP (A-n32-k5)
 * 时间预算：1s, 10s, 60s
 * 输出：CSV (instance,config,seed,obj,penalty,time_ms,gap_pct,generations,stop_reason)
 *
 * 用法：./gpu [all]
 */
#include "bench_common.cuh"

static void run_tsp_instances() {
    TSPInstance instances[] = {
        {"eil51",   eil51_coords,   EIL51_N,   426.0f},
        {"kroA100", kroA100_coords, KROA100_N, 21282.0f},
        {"ch150",   CH150_coords,   CH150_N,   6528.0f},
    };
    float time_budgets[] = {1.0f, 10.0f, 60.0f};

    for (auto& inst : instances) {
        fprintf(stderr, "  [e1] TSP %s (n=%d)\n", inst.name, inst.n);
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

static void run_vrp_instances() {
    fprintf(stderr, "  [e1] VRP A-n32-k5\n");
    float dist[AN32K5_NODES * AN32K5_NODES];
    compute_euc2d_dist(dist, an32k5_coords, AN32K5_NODES);

    float time_budgets[] = {1.0f, 10.0f, 60.0f};
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
    bench_init();
    bench_csv_header();
    run_tsp_instances();
    run_vrp_instances();
    fprintf(stderr, "\n[e1] GPU side completed.\n");
    return 0;
}
