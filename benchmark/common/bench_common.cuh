#pragma once
/**
 * bench_common.cuh — 所有 GPU benchmark 实验共用的工具代码
 *
 * 包含：GPU warmup、CSV 输出、距离计算、配置工厂、TSP 实例坐标数据
 */

#include "solver.cuh"
#include "tsp.cuh"
#include "tsp_large.cuh"
#include "tsp_xlarge.cuh"
#include "knapsack.cuh"
#include "assignment.cuh"
#include "schedule.cuh"
#include "vrp.cuh"
#include "vrptw.cuh"
#include "load_balance.cuh"
#include "graph_color.cuh"
#include "bin_packing.cuh"
#include "qap.cuh"
#include "jsp.cuh"
#include "tsplib_data.h"

#include <cmath>
#include <cstdio>
#include <cstring>

// ============================================================
// 常量
// ============================================================
static const unsigned BENCH_SEEDS[] = {42, 123, 456, 789, 2024};
static const int BENCH_NUM_SEEDS = 5;

// ============================================================
// GPU 预热
// ============================================================
static void bench_warmup() {
    float dd[25] = {};
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            dd[i * 5 + j] = (i == j) ? 0 : 10;
    auto p = TSPProblem::create(dd, 5);
    SolverConfig c;
    c.pop_size = 64; c.max_gen = 10; c.seed = 1; c.verbose = false;
    solve(p, c);
    p.destroy();
}

static void bench_print_gpu_info() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    fprintf(stderr, "GPU: %s (SM=%d, Shared=%zuKB, Compute=%d.%d)\n",
            prop.name, prop.multiProcessorCount,
            prop.sharedMemPerBlock / 1024, prop.major, prop.minor);
}

static void bench_init() {
    bench_print_gpu_info();
    fprintf(stderr, "Warming up GPU...\n");
    bench_warmup();
    fprintf(stderr, "Warmup done.\n\n");
}

// ============================================================
// CSV 输出
// ============================================================
static void bench_csv_header() {
    printf("instance,config,seed,obj,penalty,time_ms,gap_pct,generations,stop_reason\n");
    fflush(stdout);
}

static float bench_calc_gap(float obj, float known_optimal) {
    if (known_optimal == 0.0f) return 0.0f;
    if (known_optimal > 0.0f)
        return (obj - known_optimal) / known_optimal * 100.0f;
    float opt_abs = -known_optimal;
    return (opt_abs - obj) / opt_abs * 100.0f;
}

template<typename Result>
static void bench_print_row(const char* instance, const char* config,
                            unsigned seed, const Result& result,
                            float known_optimal) {
    float obj = result.best_solution.objectives[0];
    float pen = result.best_solution.penalty;
    float gap = bench_calc_gap(obj, known_optimal);
    const char* reason = (result.stop_reason == StopReason::TimeLimit)  ? "time" :
                         (result.stop_reason == StopReason::Stagnation) ? "stag" : "gen";
    printf("%s,%s,%u,%.2f,%.2f,%.1f,%.2f,%d,%s\n",
           instance, config, seed, obj, pen, result.elapsed_ms, gap,
           result.generations, reason);
    fflush(stdout);
}

// ============================================================
// 通用求解模板
// ============================================================
template<typename Problem>
void bench_run(const char* instance, const char* config_name,
               Problem& prob, const SolverConfig& cfg,
               float known_optimal, int num_seeds = BENCH_NUM_SEEDS) {
    for (int s = 0; s < num_seeds; s++) {
        SolverConfig c = cfg;
        c.seed = BENCH_SEEDS[s];
        c.verbose = false;
        auto result = solve(prob, c);
        bench_print_row(instance, config_name, BENCH_SEEDS[s], result, known_optimal);
    }
}

template<typename CreateFn>
void bench_run_recreate(const char* instance, const char* config_name,
                        CreateFn create_fn, const SolverConfig& cfg,
                        float known_optimal, int num_seeds = BENCH_NUM_SEEDS) {
    for (int s = 0; s < num_seeds; s++) {
        SolverConfig c = cfg;
        c.seed = BENCH_SEEDS[s];
        c.verbose = false;
        auto prob = create_fn();
        auto result = solve(prob, c);
        bench_print_row(instance, config_name, BENCH_SEEDS[s], result, known_optimal);
        prob.destroy();
    }
}

// ============================================================
// EUC_2D 距离计算
// ============================================================
static void compute_euc2d_dist(float* dist, const float coords[][2], int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            dist[i * n + j] = roundf(sqrtf(dx * dx + dy * dy));
        }
}

// ============================================================
// 配置工厂
// ============================================================
static SolverConfig make_default_config(int gen = 5000) {
    SolverConfig c;
    c.pop_size = 0;
    c.max_gen = gen;
    c.verbose = false;
    c.sa_temp_init = 50.0f;
    c.sa_alpha = 0.999f;
    c.num_islands = 0;
    c.migrate_interval = 50;
    c.migrate_strategy = MigrateStrategy::Hybrid;
    c.crossover_rate = 0.1f;
    c.use_aos = true;
    return c;
}

static SolverConfig make_timed_config(float seconds) {
    SolverConfig c = make_default_config(999999);
    c.time_limit_sec = seconds;
    c.stagnation_limit = 0;
    return c;
}

static SolverConfig make_hc_config(int gen = 10000) {
    SolverConfig c;
    c.pop_size = 0;
    c.max_gen = gen;
    c.verbose = false;
    return c;
}

// ============================================================
// TSP 实例坐标数据（内嵌小实例，大实例来自 tsplib_data.h）
// ============================================================
static const int EIL51_N = 51;
static const float eil51_coords[EIL51_N][2] = {
    {37,52},{49,49},{52,64},{20,26},{40,30},{21,47},{17,63},{31,62},{52,33},
    {51,21},{42,41},{31,32},{ 5,25},{12,42},{36,16},{52,41},{27,23},{17,33},
    {13,13},{57,58},{62,42},{42,57},{16,57},{ 8,52},{ 7,38},{27,68},{30,48},
    {43,67},{58,48},{58,27},{37,69},{38,46},{46,10},{61,33},{62,63},{63,69},
    {32,22},{45,35},{59,15},{ 5, 6},{10,17},{21,10},{ 5,64},{30,15},{39,10},
    {32,39},{25,32},{25,55},{48,28},{56,37},{30,40}
};

static const int KROA100_N = 100;
static const float kroA100_coords[KROA100_N][2] = {
    {1380,939},{2848,96},{3510,1671},{457,334},{3888,666},{984,965},{2721,1482},
    {1286,525},{2716,1432},{738,1325},{1251,1832},{2728,1698},{3815,169},{3683,1533},
    {1247,1945},{123,862},{1234,1946},{252,1240},{611,673},{2576,1676},{928,1700},
    {53,857},{1807,1711},{274,1420},{2574,946},{178,24},{2678,1825},{1795,962},
    {3384,1498},{3520,1079},{1256,61},{1424,1728},{3913,192},{3085,1528},{2573,1969},
    {463,1670},{3875,598},{298,1513},{3479,821},{2542,236},{3955,1743},{1323,280},
    {3447,1830},{2936,337},{1621,1830},{3373,1646},{1393,1368},{3874,1318},{938,955},
    {3022,474},{2482,1183},{3854,923},{376,825},{2519,135},{2945,1622},{953,268},
    {2628,1479},{2097,981},{890,1846},{2139,1806},{2421,1007},{2290,1810},{1115,1052},
    {2588,302},{327,265},{241,341},{1917,687},{2991,792},{2573,599},{19,674},
    {3911,1673},{872,1559},{2863,558},{929,1766},{839,620},{3893,102},{2178,1619},
    {3822,899},{378,1048},{1178,100},{2599,901},{3416,143},{2961,1605},{611,1384},
    {3113,885},{2597,1830},{2586,1286},{161,906},{1429,134},{742,1025},{1625,1651},
    {1187,706},{1787,1009},{22,987},{3640,43},{3756,882},{776,392},{1724,1642},
    {198,1810},{3950,1558}
};

// VRP A-n32-k5 数据
static const int AN32K5_N = 31;
static const int AN32K5_NODES = 32;
static const float an32k5_coords[AN32K5_NODES][2] = {
    {82,76},
    {96,44},{50,5},{49,8},{13,7},{29,89},{58,30},{84,39},{14,24},{2,39},
    {3,82},{5,10},{98,52},{84,25},{61,59},{1,65},{88,51},{91,2},{19,32},
    {93,3},{50,93},{98,14},{5,42},{42,9},{61,62},{9,97},{80,55},{57,69},
    {23,15},{20,70},{85,60},{98,5}
};
static const float an32k5_demands[AN32K5_N] = {
    19,21,6,19,7,12,16,6,16,8,14,21,16,3,22,18,19,1,24,8,12,4,8,24,24,2,20,15,2,14,9
};

// TSP 实例描述结构
struct TSPInstance {
    const char* name;
    const float (*coords)[2];
    int n;
    float optimal;
};

static TSPInstance ALL_TSP_INSTANCES[] = {
    {"eil51",   eil51_coords,   EIL51_N,   426.0f},
    {"kroA100", kroA100_coords, KROA100_N, 21282.0f},
    {"ch150",   CH150_coords,   CH150_N,   6528.0f},
    {"tsp225",  TSP225_coords,  TSP225_N,  3916.0f},
    {"lin318",  LIN318_coords,  LIN318_N,  42029.0f},
    {"pcb442",  PCB442_coords,  PCB442_N,  50778.0f},
};
static const int NUM_TSP_INSTANCES = sizeof(ALL_TSP_INSTANCES) / sizeof(ALL_TSP_INSTANCES[0]);

// 根据 N 选择合适的 TSP Problem 类型并运行
template<typename Fn>
void bench_run_tsp(const char* instance, const char* config, int n,
                   float* dist, const SolverConfig& cfg, float optimal,
                   int num_seeds = BENCH_NUM_SEEDS) {
    if (n <= 64) {
        bench_run_recreate(instance, config,
            [&]() { return TSPProblem::create(dist, n); }, cfg, optimal, num_seeds);
    } else if (n <= 256) {
        bench_run_recreate(instance, config,
            [&]() { return TSPLargeProblem::create(dist, n); }, cfg, optimal, num_seeds);
    } else {
        bench_run_recreate(instance, config,
            [&]() { return TSPXLargeProblem::create(dist, n); }, cfg, optimal, num_seeds);
    }
}
