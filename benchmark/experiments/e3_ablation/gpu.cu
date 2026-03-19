/**
 * E3: 消融实验 — 验证各模块的贡献
 *
 * 目的：通过 additive 和 leave-one-out 两种方式验证 SA/Islands/CX/AOS 的贡献
 * 实例：TSP kroA100+ch150 (Perm), BinPack20 (Int), GraphColor20 (Int),
 *        Schedule5x6 (Binary), JSP4x3 (Perm multiset)
 * 配置：HC → +SA → +Isl → +CX → Full, Full-noSA, Full-noIsl, Full-noCX, Full-noAOS
 * 输出：CSV
 *
 * 用法：./gpu [all]
 */
#include "bench_common.cuh"

static constexpr int ABLATION_GEN = 10000;

struct AblationConfig {
    const char* name;
    SolverConfig cfg;
};

static int build_configs(AblationConfig* out) {
    int count = 0;
    SolverConfig full = make_default_config(ABLATION_GEN);

    // Additive
    SolverConfig hc = make_hc_config(ABLATION_GEN);

    SolverConfig sa = make_hc_config(ABLATION_GEN);
    sa.sa_temp_init = 50.0f;
    sa.sa_alpha = 0.999f;

    SolverConfig sa_isl = sa;
    sa_isl.num_islands = 4;
    sa_isl.migrate_interval = 50;
    sa_isl.migrate_strategy = MigrateStrategy::Hybrid;

    SolverConfig sa_isl_cx = sa_isl;
    sa_isl_cx.crossover_rate = 0.1f;

    // Leave-one-out
    SolverConfig no_sa = full; no_sa.sa_temp_init = 0.0f;
    SolverConfig no_isl = full; no_isl.num_islands = 1;
    SolverConfig no_cx = full; no_cx.crossover_rate = 0.0f;
    SolverConfig no_aos = full; no_aos.use_aos = false;

    out[count++] = {"HC",          hc};
    out[count++] = {"SA",          sa};
    out[count++] = {"SA_Isl4",     sa_isl};
    out[count++] = {"SA_Isl4_CX",  sa_isl_cx};
    out[count++] = {"Full",        full};
    out[count++] = {"Full_noSA",   no_sa};
    out[count++] = {"Full_noIsl",  no_isl};
    out[count++] = {"Full_noCX",   no_cx};
    out[count++] = {"Full_noAOS",  no_aos};
    return count;
}

int main(int argc, char** argv) {
    bench_init();
    bench_csv_header();

    AblationConfig configs[16];
    int nc = build_configs(configs);

    // Part A: TSP (Permutation)
    {
        TSPInstance tsp[] = {
            {"kroA100", kroA100_coords, KROA100_N, 21282.0f},
            {"ch150",   CH150_coords,   CH150_N,   6528.0f},
        };
        for (auto& inst : tsp) {
            fprintf(stderr, "  [e3] TSP %s (n=%d)\n", inst.name, inst.n);
            float* dist = new float[inst.n * inst.n];
            compute_euc2d_dist(dist, inst.coords, inst.n);
            for (int i = 0; i < nc; i++) {
                bench_run_recreate(inst.name, configs[i].name,
                    [&]() { return TSPLargeProblem::create(dist, inst.n); },
                    configs[i].cfg, inst.optimal);
            }
            delete[] dist;
        }
    }

    // Part B: BinPacking (Integer)
    {
        fprintf(stderr, "  [e3] BinPacking20\n");
        const int N = 20;
        float weights[N] = {7,5,3,4,6,2,8,1,9,3,5,7,4,6,2,8,3,5,7,4};
        for (int i = 0; i < nc; i++) {
            bench_run_recreate("BinPack20", configs[i].name,
                [&]() { return BinPackingProblem::create(weights, N, 8, 15.0f); },
                configs[i].cfg, 0.0f);
        }
    }

    // Part C: GraphColor (Integer)
    {
        fprintf(stderr, "  [e3] GraphColor20\n");
        const int N = 20;
        int adj[N * N] = {};
        auto edge = [&](int a, int b) { adj[a*N+b] = 1; adj[b*N+a] = 1; };
        edge(0,1); edge(0,5); edge(0,10); edge(0,15);
        edge(1,2); edge(1,6); edge(1,11);
        edge(2,3); edge(2,7); edge(2,12);
        edge(3,4); edge(3,8); edge(3,13);
        edge(4,5); edge(4,9); edge(4,14);
        edge(5,6); edge(5,16);
        edge(6,7); edge(6,17);
        edge(7,8); edge(7,18);
        edge(8,9); edge(8,19);
        edge(9,10); edge(9,15);
        edge(10,11); edge(10,16);
        edge(11,12); edge(11,17);
        edge(12,13); edge(12,18);
        edge(13,14); edge(13,19);
        edge(14,15); edge(14,16);
        edge(15,17); edge(16,18); edge(17,19); edge(18,0); edge(19,1);

        for (int i = 0; i < nc; i++) {
            bench_run_recreate("GraphColor20", configs[i].name,
                [&]() { return GraphColorProblem::create(adj, N, 4); },
                configs[i].cfg, 0.0f);
        }
    }

    // Part D: Schedule (Binary)
    {
        fprintf(stderr, "  [e3] Schedule5x6\n");
        float cost[30] = {5,3,8,4,6,2, 6,2,7,5,3,4, 4,6,3,7,5,8, 7,4,5,3,6,2, 3,5,4,6,2,7};
        for (int i = 0; i < nc; i++) {
            bench_run_recreate("Schedule5x6", configs[i].name,
                [&]() { return ScheduleProblem::create(cost, 5, 6, 3); },
                configs[i].cfg, 0.0f);
        }
    }

    // Part E: JSP (Permutation multiset)
    {
        fprintf(stderr, "  [e3] JSP4x3\n");
        int machine[12] = {0,1,2, 1,2,0, 2,0,1, 0,2,1};
        float duration[12] = {3,2,4, 4,3,2, 2,4,3, 3,2,5};
        for (int i = 0; i < nc; i++) {
            bench_run_recreate("JSP4x3_Perm", configs[i].name,
                [&]() { return JSPPermProblem::create(machine, duration, 4, 3, 3); },
                configs[i].cfg, 0.0f);
        }
    }

    fprintf(stderr, "\n[e3] Ablation completed.\n");
    return 0;
}
