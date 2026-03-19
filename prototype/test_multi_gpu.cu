/**
 * test_multi_gpu.cu - 多 GPU 协同功能测试
 * 
 * 测试内容：
 * 1. 编译检查：multi_gpu_solver.cuh 是否能正确编译
 * 2. 单 GPU 回归：num_gpus=1 时结果与普通 solve 一致
 * 3. 多 GPU 基本功能：num_gpus>1 时能正常运行并返回结果
 * 4. clone_to_device 测试：Problem 能否正确克隆到不同 GPU
 */

#include "core/multi_gpu_solver.cuh"
#include "problems/tsp.cuh"
#include "problems/vrp.cuh"
#include "problems/qap.cuh"
#include <cstdio>
#include <cmath>

// ============================================================
// 辅助函数：生成测试数据
// ============================================================

void generate_random_tsp(float* dist, int n, unsigned seed = 42) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        dist[i * n + i] = 0.0f;
        for (int j = i + 1; j < n; j++) {
            float d = 10.0f + (rand() % 1000) / 10.0f;
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
}

void generate_random_vrp(float* dist, float* demand, int n, unsigned seed = 42) {
    srand(seed);
    int stride = n + 1;
    // 生成距离矩阵（包含 depot）
    for (int i = 0; i < stride; i++) {
        dist[i * stride + i] = 0.0f;
        for (int j = i + 1; j < stride; j++) {
            float d = 10.0f + (rand() % 1000) / 10.0f;
            dist[i * stride + j] = d;
            dist[j * stride + i] = d;
        }
    }
    // 生成需求
    for (int i = 0; i < n; i++) {
        demand[i] = 5.0f + (rand() % 20);
    }
}

// ============================================================
// 测试 1: 编译检查 + 单 GPU 回归
// ============================================================

void test_single_gpu_regression() {
    printf("\n=== Test 1: Single GPU Regression ===\n");
    
    const int n = 20;
    float* h_dist = new float[n * n];
    generate_random_tsp(h_dist, n);
    
    auto prob = TSPProblem::create(h_dist, n);
    
    SolverConfig cfg;
    cfg.pop_size = 128;
    cfg.max_gen = 500;
    cfg.verbose = false;
    cfg.seed = 42;
    cfg.num_islands = 4;
    cfg.use_aos = true;
    
    // 普通 solve
    auto result1 = solve(prob, cfg);
    
    // 多 GPU solve（但 num_gpus=1）
    cfg.num_gpus = 1;
    auto result2 = solve_multi_gpu(prob, cfg);
    
    printf("  Normal solve: obj=%.2f, penalty=%.2f\n",
           result1.best_solution.objectives[0], result1.best_solution.penalty);
    printf("  Multi-GPU (n=1): obj=%.2f, penalty=%.2f\n",
           result2.best_solution.objectives[0], result2.best_solution.penalty);
    
    // 检查结果是否一致（允许小误差，因为浮点运算顺序可能不同）
    float diff = fabs(result1.best_solution.objectives[0] - result2.best_solution.objectives[0]);
    if (diff < 1.0f) {
        printf("  ✅ PASS: Results match (diff=%.4f)\n", diff);
    } else {
        printf("  ❌ FAIL: Results differ significantly (diff=%.4f)\n", diff);
    }
    
    prob.destroy();
    delete[] h_dist;
}

// ============================================================
// 测试 2: clone_to_device 功能
// ============================================================

void test_clone_to_device() {
    printf("\n=== Test 2: clone_to_device() ===\n");
    
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("  Available GPUs: %d\n", device_count);
    
    if (device_count < 2) {
        printf("  ⚠️  SKIP: Need at least 2 GPUs for this test\n");
        return;
    }
    
    const int n = 15;
    float* h_dist = new float[n * n];
    generate_random_tsp(h_dist, n);
    
    // 在 GPU 0 上创建 Problem
    CUDA_CHECK(cudaSetDevice(0));
    auto prob0 = TSPProblem::create(h_dist, n);
    
    // 克隆到 GPU 1
    auto* prob1 = prob0.clone_to_device(1);
    
    if (prob1 == nullptr) {
        printf("  ❌ FAIL: clone_to_device returned nullptr\n");
        prob0.destroy();
        delete[] h_dist;
        return;
    }
    
    printf("  ✅ PASS: clone_to_device succeeded\n");
    
    // 验证克隆的 Problem 能在 GPU 1 上运行
    CUDA_CHECK(cudaSetDevice(1));
    SolverConfig cfg;
    cfg.pop_size = 64;
    cfg.max_gen = 100;
    cfg.verbose = false;
    
    auto result = solve(*prob1, cfg);
    printf("  GPU 1 solve result: obj=%.2f, penalty=%.2f\n",
           result.best_solution.objectives[0], result.best_solution.penalty);
    
    if (result.best_solution.penalty == 0.0f) {
        printf("  ✅ PASS: Cloned problem runs correctly on GPU 1\n");
    } else {
        printf("  ❌ FAIL: Cloned problem has unexpected penalty\n");
    }
    
    prob0.destroy();
    prob1->destroy();
    delete prob1;
    delete[] h_dist;
}

// ============================================================
// 测试 3: 多 GPU 协同基本功能
// ============================================================

void test_multi_gpu_basic() {
    printf("\n=== Test 3: Multi-GPU Basic Functionality ===\n");
    
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count < 2) {
        printf("  ⚠️  SKIP: Need at least 2 GPUs for this test\n");
        return;
    }
    
    const int n = 25;
    float* h_dist = new float[n * n];
    generate_random_tsp(h_dist, n);
    
    auto prob = TSPProblem::create(h_dist, n);
    
    SolverConfig cfg;
    cfg.pop_size = 128;
    cfg.time_limit_sec = 5.0f;  // 5 秒时间限制
    cfg.verbose = true;
    cfg.seed = 42;
    cfg.num_islands = 4;
    cfg.use_aos = true;
    
    // 多 GPU 求解
    cfg.num_gpus = std::min(2, device_count);
    cfg.multi_gpu_interval_sec = 2.0f;  // 每 2 秒交换一次
    cfg.multi_gpu_inject_mode = MultiGpuInjectMode::HalfIslands;
    
    printf("  Running with %d GPUs...\n", cfg.num_gpus);
    auto result = solve_multi_gpu(prob, cfg);
    
    printf("  Result: obj=%.2f, penalty=%.2f\n",
           result.best_solution.objectives[0], result.best_solution.penalty);
    
    if (result.best_solution.penalty == 0.0f && result.best_solution.objectives[0] > 0.0f) {
        printf("  ✅ PASS: Multi-GPU solve completed successfully\n");
    } else {
        printf("  ❌ FAIL: Multi-GPU solve returned invalid result\n");
    }
    
    prob.destroy();
    delete[] h_dist;
}

// ============================================================
// 测试 4: VRP 多 GPU 测试
// ============================================================

void test_multi_gpu_vrp() {
    printf("\n=== Test 4: Multi-GPU with VRP ===\n");
    
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count < 2) {
        printf("  ⚠️  SKIP: Need at least 2 GPUs for this test\n");
        return;
    }
    
    const int n = 20;
    int stride = n + 1;
    float* h_dist = new float[stride * stride];
    float* h_demand = new float[n];
    generate_random_vrp(h_dist, h_demand, n);
    
    auto prob = VRPProblem::create(h_dist, h_demand, n, 100.0f, 5, 5);
    
    SolverConfig cfg;
    cfg.pop_size = 128;
    cfg.time_limit_sec = 5.0f;
    cfg.verbose = true;
    cfg.seed = 42;
    cfg.num_islands = 4;
    cfg.use_aos = true;
    
    cfg.num_gpus = std::min(2, device_count);
    cfg.multi_gpu_interval_sec = 2.0f;
    cfg.multi_gpu_inject_mode = MultiGpuInjectMode::AllIslands;
    
    printf("  Running VRP with %d GPUs...\n", cfg.num_gpus);
    auto result = solve_multi_gpu(prob, cfg);
    
    printf("  Result: obj=%.2f, penalty=%.2f\n",
           result.best_solution.objectives[0], result.best_solution.penalty);
    
    if (result.best_solution.objectives[0] > 0.0f) {
        printf("  ✅ PASS: Multi-GPU VRP solve completed\n");
    } else {
        printf("  ❌ FAIL: Multi-GPU VRP solve returned invalid result\n");
    }
    
    prob.destroy();
    delete[] h_dist;
    delete[] h_demand;
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("╔═══════════════════════════════════════════════════╗\n");
    printf("║  Multi-GPU Solver Test Suite                     ║\n");
    printf("╚═══════════════════════════════════════════════════╝\n");
    
    // 检查 GPU 可用性
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("\nSystem Info:\n");
    printf("  Available GPUs: %d\n", device_count);
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("  GPU %d: %s (SM %d.%d, %.1f GB)\n",
               i, prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    
    // 运行测试
    test_single_gpu_regression();
    test_clone_to_device();
    test_multi_gpu_basic();
    test_multi_gpu_vrp();
    
    printf("\n╔═══════════════════════════════════════════════════╗\n");
    printf("║  All tests completed                              ║\n");
    printf("╚═══════════════════════════════════════════════════╝\n");
    
    return 0;
}
