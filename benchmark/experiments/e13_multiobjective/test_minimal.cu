#include "solver.cuh"
#include "bi_objective_vrp.cuh"
#include <cstdio>

int main() {
    printf("开始测试...\n");
    fflush(stdout);
    
    // 简单的 3x3 距离矩阵（包含 depot）
    float dist[9] = {
        0, 10, 20,
        10, 0, 15,
        20, 15, 0
    };
    
    float demand[2] = {5, 5};
    
    printf("创建 Problem...\n");
    fflush(stdout);
    
    auto prob = BiObjectiveVRP::create(dist, demand, 2, 10.0f, 2);
    
    printf("Problem 创建成功\n");
    printf("配置 Solver...\n");
    fflush(stdout);
    
    SolverConfig cfg;
    cfg.pop_size = 32;
    cfg.max_gen = 100;
    cfg.verbose = true;
    cfg.seed = 42;
    
    printf("开始求解...\n");
    fflush(stdout);
    
    auto result = solve(prob, cfg);
    
    printf("求解完成！\n");
    printf("距离: %.2f, 车辆数: %.0f\n", 
           result.best_solution.objectives[0],
           result.best_solution.objectives[1]);
    
    prob.destroy();
    return 0;
}
