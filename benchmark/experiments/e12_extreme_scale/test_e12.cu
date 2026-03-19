#include "solver.cuh"
#include "multi_gpu_solver.cuh"
#include "extreme_tsp.cuh"
#include "extreme_vrp.cuh"
#include <cstdio>
#include <vector>

void generate_random_tsp(float* dist, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        dist[i * n + i] = 0.0f;
        for (int j = i + 1; j < n; j++) {
            float d = 10.0f + (rand() % 10000) / 10.0f;
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
}

void generate_random_vrp(float* dist, float* demand, int n, unsigned seed) {
    srand(seed);
    int stride = n + 1;
    for (int i = 0; i < stride; i++) {
        dist[i * stride + i] = 0.0f;
        for (int j = i + 1; j < stride; j++) {
            float d = 10.0f + (rand() % 10000) / 10.0f;
            dist[i * stride + j] = d;
            dist[j * stride + i] = d;
        }
    }
    for (int i = 0; i < n; i++) {
        demand[i] = 5.0f + (rand() % 20);
    }
}

int main() {
    printf("==============================================\n");
    printf("E12: 极大规模多 GPU 实验\n");
    printf("==============================================\n\n");
    
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    printf("检测到 %d 个 GPU\n\n", num_gpus);
    
    const int num_runs = 3;
    
    // ========== TSP n=2000 ==========
    printf("[TSP n=2000]\n");
    printf("  工作集: 2000×2000×4 = 16 MB\n");
    printf("  预估种群: ~16 (L2=6MB)\n\n");
    
    int n_tsp = 2000;
    float* h_dist_tsp = new float[n_tsp * n_tsp];
    printf("  生成数据...\n");
    generate_random_tsp(h_dist_tsp, n_tsp, 12345);
    
    printf("  创建 Problem...\n");
    auto prob_tsp = ExtremeTSPProblem::create(h_dist_tsp, n_tsp);
    
    SolverConfig cfg;
    cfg.pop_size = 0;
    cfg.max_gen = 5000;
    cfg.verbose = false;
    cfg.num_islands = 16;
    cfg.use_aos = true;
    cfg.sa_temp_init = 50.0f;
    cfg.use_cuda_graph = true;
    
    // 单GPU
    printf("  单GPU: ");
    std::vector<float> single_results;
    for (int run = 0; run < num_runs; run++) {
        cfg.seed = 42 + run * 100;
        auto result = solve(prob_tsp, cfg);
        single_results.push_back(result.best_solution.objectives[0]);
        printf("%.1f ", result.best_solution.objectives[0]);
    }
    float avg_single = 0;
    for (float v : single_results) avg_single += v;
    avg_single /= num_runs;
    printf("→ %.2f\n", avg_single);
    
    // 多GPU
    if (num_gpus >= 2) {
        printf("  %dGPU: ", num_gpus);
        std::vector<float> multi_results;
        cfg.num_gpus = num_gpus;
        for (int run = 0; run < num_runs; run++) {
            cfg.seed = 42 + run * 100;
            auto result = solve_multi_gpu(prob_tsp, cfg);
            multi_results.push_back(result.best_solution.objectives[0]);
            printf("%.1f ", result.best_solution.objectives[0]);
        }
        float avg_multi = 0;
        for (float v : multi_results) avg_multi += v;
        avg_multi /= num_runs;
        float improvement = (avg_single - avg_multi) / avg_single * 100;
        printf("→ %.2f (%.2f%%)\n", avg_multi, improvement);
    }
    
    prob_tsp.destroy();
    delete[] h_dist_tsp;
    
    printf("\n");
    
    // ========== VRP n=1000, 160 vehicles ==========
    printf("[VRP n=1000, vehicles=160]\n");
    printf("  配置: D1=160, D2=128, Solution=80KB\n");
    printf("  需求: 5-24 (平均14.5), 容量: 100\n");
    printf("  理论需要车辆: 146, 实际: 160 (留14辆余量)\n");
    printf("  工作集: 1001×1001×4 = 4 MB\n\n");
    
    int n_vrp = 1000;
    int num_veh = 160;
    float* h_dist_vrp = new float[(n_vrp+1) * (n_vrp+1)];
    float* h_demand_vrp = new float[n_vrp];
    
    printf("  生成数据...\n");
    generate_random_vrp(h_dist_vrp, h_demand_vrp, n_vrp, 12345);
    
    printf("  创建 Problem...\n");
    auto prob_vrp = ExtremeVRPProblem::create(h_dist_vrp, h_demand_vrp, n_vrp, 100.0f, num_veh, num_veh);
    
    cfg.max_gen = 5000;
    
    // 单GPU
    printf("  单GPU: ");
    single_results.clear();
    for (int run = 0; run < num_runs; run++) {
        cfg.seed = 42 + run * 100;
        auto result = solve(prob_vrp, cfg);
        single_results.push_back(result.best_solution.objectives[0]);
        printf("%.1f ", result.best_solution.objectives[0]);
    }
    avg_single = 0;
    for (float v : single_results) avg_single += v;
    avg_single /= num_runs;
    printf("→ %.2f\n", avg_single);
    
    // 多GPU
    if (num_gpus >= 2) {
        printf("  %dGPU: ", num_gpus);
        std::vector<float> multi_results;
        cfg.num_gpus = num_gpus;
        for (int run = 0; run < num_runs; run++) {
            cfg.seed = 42 + run * 100;
            auto result = solve_multi_gpu(prob_vrp, cfg);
            multi_results.push_back(result.best_solution.objectives[0]);
            printf("%.1f ", result.best_solution.objectives[0]);
        }
        float avg_multi = 0;
        for (float v : multi_results) avg_multi += v;
        avg_multi /= num_runs;
        float improvement = (avg_single - avg_multi) / avg_single * 100;
        printf("→ %.2f (%.2f%%)\n", avg_multi, improvement);
    }
    
    prob_vrp.destroy();
    delete[] h_dist_vrp;
    delete[] h_demand_vrp;
    
    printf("\n==============================================\n");
    printf("E12 极大规模实验完成\n");
    printf("==============================================\n");
    
    return 0;
}
