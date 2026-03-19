// GenSolver 性能诊断专用 benchmark
// 目的：精确分解单个问题实例的时间构成
//
// 实验设计：
//   1. 固定单个问题（CVRP10），固定 seed=42，max_gen=2000
//   2. 变量：migrate_interval = 50, 100, 200, 500, 2000
//   3. 对照组：关闭 AOS (use_aos=false)，batch=2000（纯 GPU 计算基线）
//   4. 每组跑 3 次取中位数，消除噪声
//
// 输出 CSV：config,run,time_ms,obj,gap_pct,generations
// 配合 nvprof 使用时只跑单次（避免 profiling 开销叠加）

#include "solver.cuh"
#include "tsp.cuh"
#include "vrp.cuh"
#include "knapsack.cuh"
#include "schedule.cuh"
#include "qap.cuh"

#include <cstdio>
#include <cstring>
#include <cmath>

static void warmup() {
    float dist[25] = {0,3,6,5,7, 3,0,3,4,5, 6,3,0,5,4, 5,4,5,0,3, 7,5,4,3,0};
    auto p = TSPProblem::create(dist, 5);
    SolverConfig c;
    c.pop_size = 64; c.max_gen = 10; c.seed = 1; c.verbose = false;
    solve(p, c);
    p.destroy();
}

static SolverConfig make_config(int batch, bool aos, int aos_interval = 1) {
    SolverConfig c;
    c.pop_size = 0;
    c.max_gen = 2000;
    c.verbose = false;
    c.sa_temp_init = 50.0f;
    c.sa_alpha = 0.999f;
    c.num_islands = 0;
    c.migrate_interval = batch;
    c.migrate_strategy = MigrateStrategy::Hybrid;
    c.crossover_rate = 0.1f;
    c.use_aos = aos;
    c.aos_update_interval = aos_interval;
    c.seed = 42;
    return c;
}

struct TestProblem {
    const char* name;
    float known_optimal;
};

template<typename Problem>
static void run_single(const char* config_name, Problem& prob,
                       SolverConfig cfg, float known_opt, int repeats) {
    for (int r = 0; r < repeats; r++) {
        cfg.seed = 42 + r * 111;
        auto result = solve(prob, cfg);
        float obj = result.best_solution.objectives[0];
        float gap = (known_opt != 0.0f)
            ? (obj - known_opt) / fabsf(known_opt) * 100.0f
            : obj;
        printf("%s,%d,%.1f,%.2f,%.2f,%d\n",
               config_name, r, result.elapsed_ms, obj, gap, result.generations);
        fflush(stdout);
    }
}

int main(int argc, char** argv) {
    // argv[1]: "all" | "baseline" (batch2000_noaos only) | "default" (batch50_aos only)
    const char* mode = (argc > 1) ? argv[1] : "all";
    bool only_baseline = (strcmp(mode, "baseline") == 0);
    bool only_default  = (strcmp(mode, "default") == 0);
    int repeats = (only_baseline || only_default) ? 1 : 3;

    {
        int device;
        cudaDeviceProp prop;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        fprintf(stderr, "GPU: %s (SM=%d, Compute=%d.%d)\n",
                prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    }
    warmup();

    printf("config,run,time_ms,obj,gap_pct,generations\n");
    fflush(stdout);

    // === 测试问题：CVRP10（中等复杂度，kernel 时间 ~600ms）===
    const int N = 10, NN = N + 1;
    float coords[NN][2] = {
        {50,50},{60,50},{70,50},{80,50},{50,60},
        {50,70},{50,80},{40,50},{30,50},{50,40},{50,30}
    };
    float demands[N] = {5,4,6,5,4,6,5,4,5,6};
    float dist[NN * NN];
    for (int i = 0; i < NN; i++)
        for (int j = 0; j < NN; j++) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            dist[i * NN + j] = roundf(sqrtf(dx * dx + dy * dy));
        }

    if (only_default) {
        // nvprof 专用：只跑默认配置（batch=50, AOS=on）
        fprintf(stderr, "\n=== CVRP10: default config (batch=50, AOS=on) ===\n");
        auto prob = VRPProblem::create(dist, demands, N, 15.0f, 4, 4);
        run_single("batch50_aos", prob, make_config(50, true), 200.0f, 1);
        prob.destroy();
        return 0;
    }

    if (only_baseline) {
        // nvprof 专用：只跑纯 GPU 基线（batch=2000, AOS=off）
        fprintf(stderr, "\n=== CVRP10: baseline (batch=2000, AOS=off) ===\n");
        auto prob = VRPProblem::create(dist, demands, N, 15.0f, 4, 4);
        run_single("batch2000_noaos", prob, make_config(2000, false), 200.0f, 1);
        prob.destroy();
        return 0;
    }

    // === 完整实验 ===
    fprintf(stderr, "\n=== CVRP10: batch size comparison ===\n");

    // 实验组 1: 不同 batch size（AOS=on）
    {
        int batches[] = {50, 100, 200, 500, 2000};
        for (int b : batches) {
            char name[64];
            snprintf(name, sizeof(name), "batch%d_aos", b);
            fprintf(stderr, "  %s ...\n", name);
            auto prob = VRPProblem::create(dist, demands, N, 15.0f, 4, 4);
            run_single(name, prob, make_config(b, true), 200.0f, repeats);
            prob.destroy();
        }
    }

    // 实验组 2: 不同 batch size（AOS=off）
    {
        int batches[] = {50, 200, 2000};
        for (int b : batches) {
            char name[64];
            snprintf(name, sizeof(name), "batch%d_noaos", b);
            fprintf(stderr, "  %s ...\n", name);
            auto prob = VRPProblem::create(dist, demands, N, 15.0f, 4, 4);
            run_single(name, prob, make_config(b, false), 200.0f, repeats);
            prob.destroy();
        }
    }

    // 实验组 3: AOS 降频
    {
        int intervals[] = {1, 5, 10};
        for (int iv : intervals) {
            char name[64];
            snprintf(name, sizeof(name), "batch50_aosint%d", iv);
            fprintf(stderr, "  %s ...\n", name);
            auto prob = VRPProblem::create(dist, demands, N, 15.0f, 4, 4);
            run_single(name, prob, make_config(50, true, iv), 200.0f, repeats);
            prob.destroy();
        }
    }

    // === Schedule3x4 ===
    fprintf(stderr, "\n=== Schedule3x4: batch size comparison ===\n");
    {
        float cost[12] = {5,3,8,4, 6,2,7,5, 4,6,3,7};
        int batches[] = {50, 200, 2000};
        for (int b : batches) {
            char name[64];
            snprintf(name, sizeof(name), "sched_batch%d_aos", b);
            fprintf(stderr, "  %s ...\n", name);
            auto prob = ScheduleProblem::create(cost, 3, 4, 2);
            run_single(name, prob, make_config(b, true), 0.0f, repeats);
            prob.destroy();
        }
        {
            auto prob = ScheduleProblem::create(cost, 3, 4, 2);
            fprintf(stderr, "  sched_batch2000_noaos ...\n");
            run_single("sched_batch2000_noaos", prob, make_config(2000, false), 0.0f, repeats);
            prob.destroy();
        }
    }

    fprintf(stderr, "\nAll done.\n");
    return 0;
}
