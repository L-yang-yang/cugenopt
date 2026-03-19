/**
 * E5: 通用性验证 — 12 种问题类型
 *
 * 目的：证明同一套框架能解 12 种不同编码/约束的问题
 * 实例：TSP5, Knapsack6, Assign4, Schedule3x4, CVRP10, LoadBal8,
 *        GraphColor10, BinPack8, QAP5, VRPTW8, JSP3x3_Int, JSP3x3_Perm
 * 配置：default (gen=2000)
 * 输出：CSV
 *
 * 用法：./gpu [all]
 */
#include "bench_common.cuh"

int main(int argc, char** argv) {
    bench_init();
    bench_csv_header();

    const int GEN = 2000;
    const char* cfg_name = "default_g2k";

    // 1. TSP5
    {
        float dist[25] = {0,3,6,5,7, 3,0,3,4,5, 6,3,0,5,4, 5,4,5,0,3, 7,5,4,3,0};
        auto p = TSPProblem::create(dist, 5);
        SolverConfig c = make_default_config(GEN);
        bench_run("TSP5", cfg_name, p, c, 18.0f);
        p.destroy();
    }

    // 2. Knapsack6
    {
        float w[6] = {2,3,5,7,4,6}, v[6] = {6,5,8,14,7,10};
        auto p = KnapsackProblem::create(w, v, 6, 15.0f);
        SolverConfig c = make_default_config(GEN);
        bench_run("Knapsack6", cfg_name, p, c, -30.0f);
        p.destroy();
    }

    // 3. Assignment4
    {
        float cost[16] = {9,2,7,8, 6,4,3,7, 5,8,1,8, 7,6,9,4};
        auto p = AssignmentProblem::create(cost, 4);
        SolverConfig c = make_default_config(GEN);
        bench_run("Assign4", cfg_name, p, c, 13.0f);
        p.destroy();
    }

    // 4. Schedule3x4
    {
        float cost[12] = {5,3,8,4, 6,2,7,5, 4,6,3,7};
        auto p = ScheduleProblem::create(cost, 3, 4, 2);
        SolverConfig c = make_default_config(GEN);
        bench_run("Schedule3x4", cfg_name, p, c, 0.0f);
        p.destroy();
    }

    // 5. CVRP10
    {
        const int N = 10, NN = N + 1;
        float coords[NN][2] = {
            {50,50},{60,50},{70,50},{80,50},{50,60},{50,70},{50,80},{40,50},{30,50},{50,40},{50,30}
        };
        float demands[N] = {5,4,6,5,4,6,5,4,5,6};
        float dist[NN * NN];
        for (int i = 0; i < NN; i++)
            for (int j = 0; j < NN; j++) {
                float dx = coords[i][0] - coords[j][0];
                float dy = coords[i][1] - coords[j][1];
                dist[i * NN + j] = roundf(sqrtf(dx * dx + dy * dy));
            }
        auto p = VRPProblem::create(dist, demands, N, 15.0f, 4, 4);
        SolverConfig c = make_default_config(GEN);
        bench_run("CVRP10", cfg_name, p, c, 200.0f);
        p.destroy();
    }

    // 6. LoadBalance8
    {
        float pt[8] = {5,3,8,4,6,2,7,5};
        auto p = LoadBalanceProblem::create(pt, 8, 3);
        SolverConfig c = make_default_config(GEN);
        bench_run("LoadBal8", cfg_name, p, c, 14.0f);
        p.destroy();
    }

    // 7. GraphColor10 (Petersen)
    {
        const int N = 10;
        int adj[N * N] = {};
        auto edge = [&](int a, int b) { adj[a*N+b] = 1; adj[b*N+a] = 1; };
        edge(0,1); edge(1,2); edge(2,3); edge(3,4); edge(4,0);
        edge(5,7); edge(7,9); edge(9,6); edge(6,8); edge(8,5);
        edge(0,5); edge(1,6); edge(2,7); edge(3,8); edge(4,9);
        auto p = GraphColorProblem::create(adj, N, 3);
        SolverConfig c = make_default_config(GEN);
        bench_run("GraphColor10", cfg_name, p, c, 0.0f);
        p.destroy();
    }

    // 8. BinPacking8
    {
        float w[8] = {7,5,3,4,6,2,8,1};
        auto p = BinPackingProblem::create(w, 8, 6, 10.0f);
        SolverConfig c = make_default_config(GEN);
        bench_run("BinPack8", cfg_name, p, c, 4.0f);
        p.destroy();
    }

    // 9. QAP5
    {
        float flow[25] = {0,5,2,4,1, 5,0,3,0,2, 2,3,0,0,0, 4,0,0,0,5, 1,2,0,5,0};
        float dist[25] = {0,1,2,3,4, 1,0,1,2,3, 2,1,0,1,2, 3,2,1,0,1, 4,3,2,1,0};
        auto p = QAPProblem::create(flow, dist, 5);
        SolverConfig c = make_default_config(GEN);
        bench_run("QAP5", cfg_name, p, c, 58.0f);
        p.destroy();
    }

    // 10. VRPTW8
    {
        const int N = 8, NN = N + 1;
        float coords[NN][2] = {
            {50,50},{60,50},{70,50},{50,60},{50,70},{40,50},{30,50},{50,40},{50,30}
        };
        float demands[N] = {3,5,4,6,3,5,4,5};
        float dist[NN * NN];
        for (int i = 0; i < NN; i++)
            for (int j = 0; j < NN; j++) {
                float dx = coords[i][0] - coords[j][0];
                float dy = coords[i][1] - coords[j][1];
                dist[i * NN + j] = roundf(sqrtf(dx * dx + dy * dy));
            }
        float earliest[NN] = {0, 0,10, 0,20, 0,30, 0,10};
        float latest[NN]   = {200,50,60,50,80,50,90,50,70};
        float service[NN]  = {0, 5,5,5,5,5,5,5,5};
        auto p = VRPTWProblem::create(dist, demands, earliest, latest, service, N, 15.0f, 3, 3);
        SolverConfig c = make_default_config(GEN);
        bench_run("VRPTW8", cfg_name, p, c, 0.0f);
        p.destroy();
    }

    // 11a. JSP3x3 (Integer)
    {
        int machine[9] = {0,1,2, 1,0,2, 2,1,0};
        float duration[9] = {3,2,4, 2,3,3, 4,3,1};
        auto p = JSPProblem::create(machine, duration, 3, 3, 3, 30);
        SolverConfig c = make_default_config(GEN);
        bench_run("JSP3x3_Int", cfg_name, p, c, 12.0f);
        p.destroy();
    }

    // 11b. JSP3x3 (Perm multiset)
    {
        int machine[9] = {0,1,2, 1,0,2, 2,1,0};
        float duration[9] = {3,2,4, 2,3,3, 4,3,1};
        auto p = JSPPermProblem::create(machine, duration, 3, 3, 3);
        SolverConfig c = make_default_config(GEN);
        bench_run("JSP3x3_Perm", cfg_name, p, c, 12.0f);
        p.destroy();
    }

    fprintf(stderr, "\n[e5] Generality completed.\n");
    return 0;
}
