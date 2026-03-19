#include "solver.cuh"
#include "multi_gpu_solver.cuh"
#include "large_tsp_problem.cuh"
#include "large_vrp_problem.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>

// 生成随机TSP实例
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

// 生成随机VRP实例
void generate_random_vrp(float* dist, float* demand, int n, unsigned seed) {
    srand(seed);
    int stride = n + 1;
    // 距离矩阵（包含depot）
    for (int i = 0; i < stride; i++) {
        dist[i * stride + i] = 0.0f;
        for (int j = i + 1; j < stride; j++) {
            float d = 10.0f + (rand() % 10000) / 10.0f;
            dist[i * stride + j] = d;
            dist[j * stride + i] = d;
        }
    }
    // 需求
    for (int i = 0; i < n; i++) {
        demand[i] = 5.0f + (rand() % 20);
    }
}

int main() {
    printf("==============================================\n");
    printf("E10: 大规模问题实验 (TSP & VRP)\n");
    printf("==============================================\n\n");
    
    // 检测可用GPU数量
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    printf("检测到 %d 个 GPU\n\n", num_gpus);
    
    const int num_runs = 5;
    
    // ========== TSP 大规模测试 ==========
    printf("实验 1: TSP 大规模测试\n");
    printf("----------------------------------------------\n");
    
    std::vector<int> tsp_sizes = {100, 200, 300, 400, 500};
    
    for (int n : tsp_sizes) {
        printf("\n[TSP n=%d]\n", n);
        
        // 生成实例
        float* h_dist = new float[n * n];
        generate_random_tsp(h_dist, n, 12345);
        auto prob = LargeTSPProblem::create(h_dist, n);
        
        // 配置
        SolverConfig cfg;
        cfg.pop_size = 0;  // 自适应
        cfg.max_gen = 10000;
        cfg.verbose = false;
        cfg.num_islands = 16;
        cfg.use_aos = true;
        cfg.sa_temp_init = 50.0f;
        cfg.use_cuda_graph = true;
        
        // 单GPU测试
        printf("  单GPU (5 runs): ");
        std::vector<float> single_gpu_results;
        for (int run = 0; run < num_runs; run++) {
            cfg.seed = 42 + run * 100;
            auto result = solve(prob, cfg);
            single_gpu_results.push_back(result.best_solution.objectives[0]);
            printf("%.1f ", result.best_solution.objectives[0]);
        }
        float avg_single = 0;
        for (float v : single_gpu_results) avg_single += v;
        avg_single /= num_runs;
        printf(" → 平均: %.2f\n", avg_single);
        
        // 多GPU测试（如果可用）
        if (num_gpus >= 2) {
            printf("  多GPU (%d GPUs, 5 runs): ", num_gpus);
            std::vector<float> multi_gpu_results;
            cfg.num_gpus = num_gpus;
            for (int run = 0; run < num_runs; run++) {
                cfg.seed = 42 + run * 100;
                auto result = solve_multi_gpu(prob, cfg);
                multi_gpu_results.push_back(result.best_solution.objectives[0]);
                printf("%.1f ", result.best_solution.objectives[0]);
            }
            float avg_multi = 0;
            for (float v : multi_gpu_results) avg_multi += v;
            avg_multi /= num_runs;
            float improvement = (avg_single - avg_multi) / avg_single * 100;
            printf(" → 平均: %.2f (%.2f%%)\n", avg_multi, improvement);
        }
        
        prob.destroy();
        delete[] h_dist;
    }
    
    // ========== VRP 大规模测试 ==========
    printf("\n\n实验 2: VRP 大规模测试\n");
    printf("----------------------------------------------\n");
    
    std::vector<int> vrp_sizes = {50, 100, 150, 200};
    
    for (int n : vrp_sizes) {
        printf("\n[VRP n=%d]\n", n);
        
        // 生成实例
        float* h_dist = new float[(n+1) * (n+1)];
        float* h_demand = new float[n];
        generate_random_vrp(h_dist, h_demand, n, 23456);
        
        int num_vehicles = (n / 20) + 1;  // 动态车辆数
        float capacity = 150.0f;
        auto prob = LargeVRPProblem::create(h_dist, h_demand, n, capacity, num_vehicles, num_vehicles + 4);
        
        // 配置
        SolverConfig cfg;
        cfg.pop_size = 0;  // 自适应
        cfg.max_gen = 10000;
        cfg.verbose = false;
        cfg.num_islands = 16;
        cfg.use_aos = true;
        cfg.sa_temp_init = 50.0f;
        cfg.use_cuda_graph = true;
        
        // 单GPU测试
        printf("  单GPU (5 runs): ");
        std::vector<float> single_gpu_results;
        for (int run = 0; run < num_runs; run++) {
            cfg.seed = 42 + run * 100;
            auto result = solve(prob, cfg);
            single_gpu_results.push_back(result.best_solution.objectives[0]);
            printf("%.1f ", result.best_solution.objectives[0]);
        }
        float avg_single = 0;
        for (float v : single_gpu_results) avg_single += v;
        avg_single /= num_runs;
        printf(" → 平均: %.2f\n", avg_single);
        
        // 多GPU测试（如果可用）
        if (num_gpus >= 2) {
            printf("  多GPU (%d GPUs, 5 runs): ", num_gpus);
            std::vector<float> multi_gpu_results;
            cfg.num_gpus = num_gpus;
            for (int run = 0; run < num_runs; run++) {
                cfg.seed = 42 + run * 100;
                auto result = solve_multi_gpu(prob, cfg);
                multi_gpu_results.push_back(result.best_solution.objectives[0]);
                printf("%.1f ", result.best_solution.objectives[0]);
            }
            float avg_multi = 0;
            for (float v : multi_gpu_results) avg_multi += v;
            avg_multi /= num_runs;
            float improvement = (avg_single - avg_multi) / avg_single * 100;
            printf(" → 平均: %.2f (%.2f%%)\n", avg_multi, improvement);
        }
        
        prob.destroy();
        delete[] h_dist;
        delete[] h_demand;
    }
    
    printf("\n==============================================\n");
    printf("实验完成！\n");
    printf("==============================================\n");
    
    return 0;
}
