/**
 * E4: 可扩展性测试 — 问题规模 vs 性能
 *
 * 目的：测试 GenSolver 在不同规模 TSP 上的 gens/s、gap、时间表现
 * 实例：TSP eil51 → pcb442 (6 个规模)
 * 时间预算：5s, 10s, 30s
 * 输出：CSV
 *
 * 用法：./gpu [all]
 */
#include "bench_common.cuh"

int main(int argc, char** argv) {
    bench_init();
    bench_csv_header();

    float time_budgets[] = {5.0f, 10.0f, 30.0f};

    for (int i = 0; i < NUM_TSP_INSTANCES; i++) {
        auto& inst = ALL_TSP_INSTANCES[i];
        fprintf(stderr, "  [e4] %s (n=%d)\n", inst.name, inst.n);

        float* dist = new float[inst.n * inst.n];
        compute_euc2d_dist(dist, inst.coords, inst.n);

        for (float t : time_budgets) {
            char cfg[64];
            snprintf(cfg, sizeof(cfg), "scale_%.0fs", t);
            SolverConfig c = make_timed_config(t);
            bench_run_tsp<void>(inst.name, cfg, inst.n, dist, c, inst.optimal);
        }
        delete[] dist;
    }

    fprintf(stderr, "\n[e4] Scalability completed.\n");
    return 0;
}
