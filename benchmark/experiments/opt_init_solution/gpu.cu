/**
 * opt_init_solution: 属性双向构造初始解 验证实验
 *
 * 对比：heuristic init（当前代码，TSP 自动注入距离矩阵构造解）
 *       vs E4 baseline 数据（纯随机初始解）
 *
 * 测试实例：eil51, lin318, pcb442
 * 时间预算：5s, 10s, 30s
 * 输出：CSV
 */
#include "bench_common.cuh"

int main(int argc, char** argv) {
    bench_init();
    bench_csv_header();

    float time_budgets[] = {5.0f, 10.0f, 30.0f};

    // eil51 — 小规模回归测试
    {
        auto& inst = ALL_TSP_INSTANCES[0]; // eil51
        float* dist = new float[inst.n * inst.n];
        compute_euc2d_dist(dist, inst.coords, inst.n);
        for (float t : time_budgets) {
            char cfg[64];
            snprintf(cfg, sizeof(cfg), "heur_%.0fs", t);
            SolverConfig c = make_timed_config(t);
            bench_run_tsp<void>(inst.name, cfg, inst.n, dist, c, inst.optimal);
        }
        delete[] dist;
    }

    // lin318 — 中大规模
    {
        auto& inst = ALL_TSP_INSTANCES[4]; // lin318
        float* dist = new float[inst.n * inst.n];
        compute_euc2d_dist(dist, inst.coords, inst.n);
        for (float t : time_budgets) {
            char cfg[64];
            snprintf(cfg, sizeof(cfg), "heur_%.0fs", t);
            SolverConfig c = make_timed_config(t);
            bench_run_tsp<void>(inst.name, cfg, inst.n, dist, c, inst.optimal);
        }
        delete[] dist;
    }

    // pcb442 — 大规模
    {
        auto& inst = ALL_TSP_INSTANCES[5]; // pcb442
        float* dist = new float[inst.n * inst.n];
        compute_euc2d_dist(dist, inst.coords, inst.n);
        for (float t : time_budgets) {
            char cfg[64];
            snprintf(cfg, sizeof(cfg), "heur_%.0fs", t);
            SolverConfig c = make_timed_config(t);
            bench_run_tsp<void>(inst.name, cfg, inst.n, dist, c, inst.optimal);
        }
        delete[] dist;
    }

    fprintf(stderr, "\n[opt_init] completed.\n");
    return 0;
}
