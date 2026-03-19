/**
 * opt_aos_interval: AOS 更新频率优化验证
 *
 * 对比 aos_update_interval = 1 (旧默认) vs 5 (新默认) vs 10
 * 测试实例：TSP eil51, ch150, lin318（覆盖小/中/大规模）
 * 配置：timed 5s, 固定 5 seeds
 * 核心指标：gens/s 和 gap
 */
#include "bench_common.cuh"

int main(int argc, char** argv) {
    bench_init();
    bench_csv_header();

    int instances[] = {0, 2, 4}; // eil51, ch150, lin318
    int intervals[] = {1, 5, 10};

    for (int ii : instances) {
        auto& inst = ALL_TSP_INSTANCES[ii];
        float* dist = new float[inst.n * inst.n];
        compute_euc2d_dist(dist, inst.coords, inst.n);

        for (int iv : intervals) {
            char cfg_name[64];
            snprintf(cfg_name, sizeof(cfg_name), "aos_iv%d", iv);

            SolverConfig c = make_timed_config(5.0f);
            c.use_aos = true;
            c.aos_update_interval = iv;

            bench_run_tsp<void>(inst.name, cfg_name, inst.n, dist, c, inst.optimal);
        }
        delete[] dist;
    }

    fprintf(stderr, "\n[opt_aos_interval] completed.\n");
    return 0;
}
