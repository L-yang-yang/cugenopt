#include "solver.cuh"
#include "multi_gpu_solver.cuh"
#include "ultra_large_tsp.cuh"
#include "ultra_large_vrp.cuh"
#include <cstdio>
#include <vector>
#include <ctime>

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
    printf("E11: 超大规模实验 (n=1000)\n");
    printf("==============================================\n\n");
    
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    printf("检测到 %d 个 GPU\n\n", num_gpus);
    
    // ========== TSP n=1000 ==========
    printf("[TSP n=1000]\n");
    printf("分配内存...\n");
    
    int n_tsp = 1000;
    float* h_dist_tsp = new float[n_tsp * n_tsp];
    printf("生成数据...\n");
    generate_random_tsp(h_dist_tsp, n_tsp, 12345);
    
    printf("创建 Problem...\n");
    auto prob_tsp = UltraLargeTSPProblem::create(h_dist_tsp, n_tsp);
    
    SolverConfig cfg;
    cfg.pop_size = 0;
    cfg.max_gen = 1000;  // 先测 1000 代
    cfg.verbose = true;
    cfg.num_islands = 16;
    cfg.use_aos = true;
    cfg.sa_temp_init = 50.0f;
    cfg.use_cuda_graph = true;
    cfg.seed = 42;
    
    printf("\n开始求解（单GPU，1000代）...\n");
    time_t start = time(nullptr);
    auto result_tsp = solve(prob_tsp, cfg);
    time_t end = time(nullptr);
    
    printf("\n结果: %.2f\n", result_tsp.best_solution.objectives[0]);
    printf("耗时: %ld 秒\n", end - start);
    printf("预估 5000 代耗时: ~%ld 秒 (%.1f 分钟)\n", 
           (end - start) * 5, (end - start) * 5.0 / 60.0);
    
    prob_tsp.destroy();
    delete[] h_dist_tsp;
    
    printf("\n");
    
    // ========== VRP n=500 (先测小一点) ==========
    printf("[VRP n=500, vehicles=25]\n");
    printf("分配内存...\n");
    
    int n_vrp = 500;
    int num_veh = 25;
    float* h_dist_vrp = new float[(n_vrp+1) * (n_vrp+1)];
    float* h_demand_vrp = new float[n_vrp];
    
    printf("生成数据...\n");
    generate_random_vrp(h_dist_vrp, h_demand_vrp, n_vrp, 12345);
    
    printf("创建 Problem...\n");
    auto prob_vrp = UltraLargeVRPProblem::create(h_dist_vrp, h_demand_vrp, n_vrp, 100.0f, num_veh, num_veh);
    
    cfg.seed = 42;
    cfg.max_gen = 1000;
    
    printf("\n开始求解（单GPU，1000代）...\n");
    start = time(nullptr);
    auto result_vrp = solve(prob_vrp, cfg);
    end = time(nullptr);
    
    printf("\n结果: %.2f\n", result_vrp.best_solution.objectives[0]);
    printf("耗时: %ld 秒\n", end - start);
    printf("预估 5000 代耗时: ~%ld 秒 (%.1f 分钟)\n", 
           (end - start) * 5, (end - start) * 5.0 / 60.0);
    
    prob_vrp.destroy();
    delete[] h_dist_vrp;
    delete[] h_demand_vrp;
    
    printf("\n==============================================\n");
    printf("E11 快速验证完成\n");
    printf("==============================================\n");
    
    return 0;
}
