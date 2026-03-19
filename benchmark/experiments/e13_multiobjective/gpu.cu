#include "solver.cuh"
#include "multi_gpu_solver.cuh"
#include "bi_objective_vrp.cuh"
#include "tri_objective_vrp.cuh"
#include "bi_objective_knapsack.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

// 确保使用 std:: 命名空间的数学函数
using std::sqrt;
using std::round;

// ============================================================
// 数据加载工具
// ============================================================

// 加载 A-n32-k5 VRP 实例（EUC_2D 格式）
struct VRPInstance {
    float* dist;
    float* demand;
    int n;
    float capacity;
    int optimal_vehicles;
    float optimal_distance;
};

VRPInstance load_an32k5() {
    // A-n32-k5 坐标（包含 depot）
    const float coords[32][2] = {
        {82,76},
        {96,44},{50,5},{49,8},{13,7},{29,89},{58,30},{84,39},{14,24},{2,39},
        {3,82},{5,10},{98,52},{84,25},{61,59},{1,65},{88,51},{91,2},{19,32},
        {93,3},{50,93},{98,14},{5,42},{42,9},{61,62},{9,97},{80,55},{57,69},
        {23,15},{20,70},{85,60},{98,5}
    };
    
    const float demands[31] = {
        19,21,6,19,7,12,16,6,16,8,14,21,16,3,22,18,19,1,24,8,12,4,8,24,24,2,20,15,2,14,9
    };
    
    VRPInstance inst;
    inst.n = 31;
    inst.capacity = 100.0f;
    inst.optimal_vehicles = 5;
    inst.optimal_distance = 784.0f;
    
    // 计算 EUC_2D 距离矩阵
    inst.dist = new float[32 * 32];
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            inst.dist[i * 32 + j] = std::round(std::sqrt(dx * dx + dy * dy));
        }
    }
    
    inst.demand = new float[31];
    for (int i = 0; i < 31; i++) {
        inst.demand[i] = demands[i];
    }
    
    return inst;
}

// 加载 knapPI_1_100 实例
struct KnapsackInstance {
    int* values;
    int* weights;
    int n;
    int capacity;
    int optimal_value;
};

KnapsackInstance load_knapsack_100() {
    const char* filename = "../../data/knapsack/knapPI_1_100.txt";
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        exit(1);
    }
    
    int n, capacity;
    file >> n >> capacity;
    
    KnapsackInstance inst;
    inst.n = n;
    inst.capacity = capacity;
    inst.optimal_value = 9147;  // 已知最优值
    
    inst.values = new int[n];
    inst.weights = new int[n];
    
    for (int i = 0; i < n; i++) {
        file >> inst.values[i] >> inst.weights[i];
    }
    
    file.close();
    return inst;
}

// ============================================================
// 实验配置
// ============================================================

struct ExperimentConfig {
    const char* name;
    CompareMode mode;
    float obj_weights[MAX_OBJ];
    int obj_priority[MAX_OBJ];
    float obj_tolerance[MAX_OBJ];
};

// Weighted 模式配置
ExperimentConfig WEIGHTED_CONFIGS[] = {
    {"W_90_10", CompareMode::Weighted, {0.9f, 0.1f}, {0, 1}, {0.0f, 0.0f}},
    {"W_70_30", CompareMode::Weighted, {0.7f, 0.3f}, {0, 1}, {0.0f, 0.0f}},
    {"W_50_50", CompareMode::Weighted, {0.5f, 0.5f}, {0, 1}, {0.0f, 0.0f}},
};

// Lexicographic 模式配置（双目标）
ExperimentConfig LEX_CONFIGS_BI[] = {
    {"L_dist_veh_t100", CompareMode::Lexicographic, {1.0f, 1.0f}, {0, 1}, {100.0f, 0.0f}},
    {"L_dist_veh_t50",  CompareMode::Lexicographic, {1.0f, 1.0f}, {0, 1}, {50.0f, 0.0f}},
    {"L_veh_dist_t0",   CompareMode::Lexicographic, {1.0f, 1.0f}, {1, 0}, {0.0f, 100.0f}},
};

// Lexicographic 模式配置（三目标）
ExperimentConfig LEX_CONFIGS_TRI[] = {
    {"L_dist_veh_max", CompareMode::Lexicographic, {1.0f, 1.0f, 1.0f}, {0, 1, 2}, {100.0f, 0.0f, 50.0f}},
    {"L_veh_dist_max", CompareMode::Lexicographic, {1.0f, 1.0f, 1.0f}, {1, 0, 2}, {0.0f, 100.0f, 50.0f}},
};

// ============================================================
// 实验运行函数
// ============================================================

template<typename Problem>
void run_experiment(const char* problem_name, Problem& prob,
                    const ExperimentConfig& exp_cfg,
                    int num_objectives,
                    bool multi_gpu = false) {
    printf("  [run_experiment] 开始\n");
    fflush(stdout);
    
    // 应用实验配置到 Problem（通过覆盖字段）
    prob.override_mode = exp_cfg.mode;
    for (int i = 0; i < num_objectives; i++) {
        prob.override_weights[i] = exp_cfg.obj_weights[i];
        prob.override_priority[i] = exp_cfg.obj_priority[i];
        prob.override_tolerance[i] = exp_cfg.obj_tolerance[i];
    }
    
    printf("  [run_experiment] 配置覆盖完成\n");
    fflush(stdout);
    
    SolverConfig cfg;
    cfg.pop_size = 64;  // 固定小规模
    cfg.max_gen = 1000;  // 固定代数
    cfg.time_limit_sec = 0.0f;  // 不使用时间限制
    cfg.verbose = true;  // 启用详细输出
    cfg.sa_temp_init = 50.0f;
    cfg.sa_alpha = 0.999f;
    cfg.num_islands = 2;  // 固定岛屿数
    cfg.migrate_interval = 50;
    cfg.crossover_rate = 0.1f;
    cfg.use_aos = true;  // 启用 AOS（测试延迟归一化）
    cfg.aos_update_interval = 5;  // 每 5 个 batch 更新一次
    cfg.use_cuda_graph = false;  // 禁用 CUDA Graph
    
    printf("  [run_experiment] SolverConfig 创建完成\n");
    fflush(stdout);
    
    const int num_runs = 1;  // 先只运行 1 次测试
    const unsigned seeds[] = {42, 123, 456, 789, 2024};
    
    printf("\n[%s] %s (mode=%s, multi_gpu=%s)\n",
           problem_name, exp_cfg.name,
           exp_cfg.mode == CompareMode::Weighted ? "Weighted" : "Lexicographic",
           multi_gpu ? "YES" : "NO");
    fflush(stdout);
    
    for (int run = 0; run < num_runs; run++) {
        printf("  [run_experiment] 开始 Run %d\n", run + 1);
        fflush(stdout);
        cfg.seed = seeds[run];
        
        SolveResult<typename Problem::Sol> result;
        if (multi_gpu) {
            cfg.num_gpus = 2;
            result = solve_multi_gpu(prob, cfg);
        } else {
            result = solve(prob, cfg);
        }
        
        printf("  Run %d (seed=%u): ", run + 1, seeds[run]);
        for (int i = 0; i < num_objectives; i++) {
            printf("obj%d=%.2f ", i, result.best_solution.objectives[i]);
        }
        printf("penalty=%.2f time=%.1fs gen=%d\n",
               result.best_solution.penalty,
               result.elapsed_ms / 1000.0f,
               result.generations);
    }
}

// ============================================================
// 主函数
// ============================================================

int main() {
    printf("==============================================\n");
    printf("E13: 多目标优化验证实验\n");
    printf("==============================================\n\n");
    fflush(stdout);
    
    // 检测 GPU
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (检测到 %d 个)\n\n", prop.name, num_gpus);
    fflush(stdout);
    
    // ========== 实验 1: 双目标 VRP (A-n32-k5) ==========
    printf("========================================\n");
    printf("实验 1: 双目标 VRP (A-n32-k5)\n");
    printf("目标: 最小化距离 + 最小化车辆数\n");
    printf("========================================\n");
    fflush(stdout);
    
    printf("加载数据...\n");
    fflush(stdout);
    VRPInstance vrp_inst = load_an32k5();
    printf("数据加载完成\n");
    fflush(stdout);
    
    // Weighted 模式测试
    printf("\n--- Weighted 模式 ---\n");
    fflush(stdout);
    
    printf("创建第一个 Problem...\n");
    fflush(stdout);
    auto prob = BiObjectiveVRP::create(vrp_inst.dist, vrp_inst.demand,
                                        vrp_inst.n, vrp_inst.capacity, 10);
    printf("Problem 创建成功，开始实验...\n");
    fflush(stdout);
    
    run_experiment("BiVRP", prob, WEIGHTED_CONFIGS[0], 2, false);
    
    printf("第一个实验完成\n");
    fflush(stdout);
    prob.destroy();
    
    // Lexicographic 模式测试
    printf("\n--- Lexicographic 模式 ---\n");
    for (int i = 0; i < 3; i++) {
        auto prob = BiObjectiveVRP::create(vrp_inst.dist, vrp_inst.demand,
                                            vrp_inst.n, vrp_inst.capacity, 10);
        run_experiment("BiVRP", prob, LEX_CONFIGS_BI[i], 2, false);
        prob.destroy();
    }
    
    // 多 GPU 验证（附加）
    if (num_gpus >= 2) {
        printf("\n--- 多 GPU 附加验证 (2×GPU) ---\n");
        
        // Weighted 验证
        auto prob_w = BiObjectiveVRP::create(vrp_inst.dist, vrp_inst.demand,
                                              vrp_inst.n, vrp_inst.capacity, 10);
        run_experiment("BiVRP_MultiGPU", prob_w, WEIGHTED_CONFIGS[1], 2, true);
        prob_w.destroy();
        
        // Lexicographic 验证
        auto prob_l = BiObjectiveVRP::create(vrp_inst.dist, vrp_inst.demand,
                                              vrp_inst.n, vrp_inst.capacity, 10);
        run_experiment("BiVRP_MultiGPU", prob_l, LEX_CONFIGS_BI[0], 2, true);
        prob_l.destroy();
    }
    
    delete[] vrp_inst.dist;
    delete[] vrp_inst.demand;
    
    // ========== 实验 2: 三目标 VRP (A-n32-k5) ==========
    printf("\n========================================\n");
    printf("实验 2: 三目标 VRP (A-n32-k5)\n");
    printf("目标: 最小化距离 + 最小化车辆数 + 最小化最大路径长度\n");
    printf("========================================\n");
    
    vrp_inst = load_an32k5();
    
    // Weighted 模式
    printf("\n--- Weighted 模式 ---\n");
    ExperimentConfig tri_weighted = {"W_60_20_20", CompareMode::Weighted, {0.6f, 0.2f, 0.2f}, {0, 1, 2}, {0.0f, 0.0f, 0.0f}};
    auto prob_tri_w = TriObjectiveVRP::create(vrp_inst.dist, vrp_inst.demand,
                                               vrp_inst.n, vrp_inst.capacity, 10);
    run_experiment("TriVRP", prob_tri_w, tri_weighted, 3, false);
    prob_tri_w.destroy();
    
    // Lexicographic 模式
    printf("\n--- Lexicographic 模式 ---\n");
    for (int i = 0; i < 2; i++) {
        auto prob_tri_l = TriObjectiveVRP::create(vrp_inst.dist, vrp_inst.demand,
                                                   vrp_inst.n, vrp_inst.capacity, 10);
        run_experiment("TriVRP", prob_tri_l, LEX_CONFIGS_TRI[i], 3, false);
        prob_tri_l.destroy();
    }
    
    delete[] vrp_inst.dist;
    delete[] vrp_inst.demand;
    
    // ========== 实验 3: 双目标 Knapsack - 暂时跳过（文件读取问题） ==========
    printf("\n========================================\n");
    printf("实验 3: 双目标 Knapsack - 跳过\n");
    printf("========================================\n");
    fflush(stdout);
    
    printf("\n==============================================\n");
    printf("E13 实验完成\n");
    printf("==============================================\n");
    
    return 0;
}
