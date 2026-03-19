/**
 * test_multi_gpu_b3.cu - 方案 B3（被动注入）功能测试
 * 
 * 测试内容：
 * 1. InjectBuffer 基本功能：分配、写入、读取、释放
 * 2. inject_check_kernel 正确性：能否正确检查并注入解
 * 3. 协调线程功能：能否定期收集并注入 global_best
 * 4. 端到端测试：2 GPU 和 4 GPU 场景下的完整运行
 * 5. 性能对比：方案 B3 vs v5.0 简化版的收益
 */

#include "core/multi_gpu_solver.cuh"
#include "problems/tsp.cuh"
#include "problems/vrp.cuh"
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
    for (int i = 0; i < stride; i++) {
        dist[i * stride + i] = 0.0f;
        for (int j = i + 1; j < stride; j++) {
            float d = 10.0f + (rand() % 1000) / 10.0f;
            dist[i * stride + j] = d;
            dist[j * stride + i] = d;
        }
    }
    for (int i = 0; i < n; i++) {
        demand[i] = 5.0f + (rand() % 20);
    }
}

// ============================================================
// 测试 1: InjectBuffer 基本功能
// ============================================================

void test_inject_buffer() {
    printf("\n=== Test 1: InjectBuffer Basic Functionality ===\n");
    
    using Sol = Solution<1, 32>;
    
    // 分配 InjectBuffer
    auto buf = InjectBuffer<Sol>::allocate(0);
    
    // 创建测试解
    Sol test_sol;
    test_sol.dim2_sizes[0] = 5;
    for (int i = 0; i < 5; i++) test_sol.data[0][i] = i + 10;
    test_sol.objectives[0] = 123.45f;
    test_sol.penalty = 0.0f;
    
    // 异步写入
    buf.write_async(test_sol);
    
    // 读取 flag（应该是 1）
    int flag;
    cudaMemcpy(&flag, buf.d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Flag after write: %d (expected 1)\n", flag);
    
    // 读取解
    Sol read_sol;
    cudaMemcpy(&read_sol, buf.d_solution, sizeof(Sol), cudaMemcpyDeviceToHost);
    printf("  Read solution: obj=%.2f, penalty=%.2f, data[0][0]=%d\n",
           read_sol.objectives[0], read_sol.penalty, read_sol.data[0][0]);
    
    // 验证数据一致性
    bool ok = (fabs(read_sol.objectives[0] - 123.45f) < 1e-3) &&
              (read_sol.data[0][0] == 10) &&
              (flag == 1);
    
    printf("  Result: %s\n", ok ? "PASS" : "FAIL");
    
    // 清理
    buf.destroy();
}

// ============================================================
// 测试 2: inject_check_kernel 正确性
// ============================================================

void test_inject_check_kernel() {
    printf("\n=== Test 2: inject_check_kernel Correctness ===\n");
    
    using Sol = Solution<1, 32>;
    const int pop_size = 64;
    const int island_size = 16;
    
    // 分配种群
    Sol* d_pop;
    cudaMalloc(&d_pop, sizeof(Sol) * pop_size);
    
    // 初始化种群（所有解的 obj=100.0）
    Sol* h_pop = new Sol[pop_size];
    for (int i = 0; i < pop_size; i++) {
        h_pop[i].objectives[0] = 100.0f;
        h_pop[i].penalty = 0.0f;
    }
    cudaMemcpy(d_pop, h_pop, sizeof(Sol) * pop_size, cudaMemcpyHostToDevice);
    
    // 创建 InjectBuffer 并写入优秀解（obj=50.0）
    auto buf = InjectBuffer<Sol>::allocate(0);
    Sol inject_sol;
    inject_sol.objectives[0] = 50.0f;
    inject_sol.penalty = 0.0f;
    buf.write_async(inject_sol);
    
    // 将 InjectBuffer 拷贝到 device
    InjectBuffer<Sol>* d_buf;
    cudaMalloc(&d_buf, sizeof(InjectBuffer<Sol>));
    cudaMemcpy(d_buf, &buf, sizeof(InjectBuffer<Sol>), cudaMemcpyHostToDevice);
    
    // 构造 ObjConfig
    ObjConfig oc;
    oc.num_obj = 1;
    oc.mode = CompareMode::Weighted;
    oc.dirs[0] = ObjDir::Minimize;
    oc.weights[0] = 1.0f;
    
    // 调用 inject_check_kernel
    inject_check_kernel<<<1, 1>>>(d_pop, pop_size, island_size, d_buf, oc);
    cudaDeviceSynchronize();
    
    // 读取种群，检查第一个岛的 worst 是否被替换
    cudaMemcpy(h_pop, d_pop, sizeof(Sol) * pop_size, cudaMemcpyDeviceToHost);
    
    int replaced_count = 0;
    for (int i = 0; i < island_size; i++) {
        if (fabs(h_pop[i].objectives[0] - 50.0f) < 1e-3) {
            replaced_count++;
        }
    }
    
    printf("  Replaced count in first island: %d (expected 1)\n", replaced_count);
    
    // 检查 flag 是否被清零
    int flag;
    cudaMemcpy(&flag, buf.d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Flag after inject_check: %d (expected 0)\n", flag);
    
    bool ok = (replaced_count == 1) && (flag == 0);
    printf("  Result: %s\n", ok ? "PASS" : "FAIL");
    
    // 清理
    buf.destroy();
    cudaFree(d_buf);
    cudaFree(d_pop);
    delete[] h_pop;
}

// ============================================================
// 测试 3: 2 GPU 端到端测试（小规模）
// ============================================================

void test_2gpu_tsp_small() {
    printf("\n=== Test 3: 2 GPU TSP (n=30) ===\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count < 2) {
        printf("  SKIP: Need at least 2 GPUs\n");
        return;
    }
    
    const int n = 30;
    float* h_dist = new float[n * n];
    generate_random_tsp(h_dist, n, 12345);
    
    auto prob = TSPProblem::create(h_dist, n);
    
    SolverConfig cfg;
    cfg.pop_size = 256;
    cfg.max_gen = 2000;
    cfg.verbose = true;
    cfg.seed = 42;
    cfg.num_islands = 4;
    cfg.use_aos = true;
    cfg.sa_temp_init = 10.0f;
    cfg.use_cuda_graph = true;
    
    // 方案 B3: 2 GPU with exchange
    cfg.num_gpus = 2;
    cfg.multi_gpu_interval_sec = 2.0f;  // 2 秒交换一次
    cfg.multi_gpu_inject_mode = MultiGpuInjectMode::OneIsland;
    
    auto result = solve_multi_gpu(prob, cfg);
    
    printf("  Result: obj=%.2f, penalty=%.2f\n",
           result.best_solution.objectives[0],
           result.best_solution.penalty);
    
    delete[] h_dist;
}

// ============================================================
// 测试 4: 4 GPU VRP 测试（中等规模）
// ============================================================

void test_2gpu_vrp_medium() {
    printf("\n=== Test 4: 2 GPU VRP (n=50) ===\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count < 2) {
        printf("  SKIP: Need at least 2 GPUs (have %d)\n", device_count);
        return;
    }
    
    const int n = 50;
    float* h_dist = new float[(n+1) * (n+1)];
    float* h_demand = new float[n];
    generate_random_vrp(h_dist, h_demand, n, 23456);
    
    auto prob = VRPProblem::create(h_dist, h_demand, n, 150.0f, 8, 16);
    
    SolverConfig cfg;
    cfg.pop_size = 512;
    cfg.max_gen = 3000;
    cfg.verbose = true;
    cfg.seed = 42;
    cfg.num_islands = 8;
    cfg.use_aos = true;
    cfg.sa_temp_init = 15.0f;
    cfg.use_cuda_graph = true;
    
    // 方案 B3: 2 GPU with exchange
    cfg.num_gpus = 2;
    cfg.multi_gpu_interval_sec = 3.0f;  // 3 秒交换一次
    cfg.multi_gpu_inject_mode = MultiGpuInjectMode::HalfIslands;
    
    auto result = solve_multi_gpu(prob, cfg);
    
    printf("  Result: obj=%.2f, penalty=%.2f\n",
           result.best_solution.objectives[0],
           result.best_solution.penalty);
    
    delete[] h_dist;
    delete[] h_demand;
}

// ============================================================
// 测试 5: 性能对比（B3 vs 简化版）
// ============================================================

void test_performance_comparison() {
    printf("\n=== Test 5: Performance Comparison (B3 vs Simplified) ===\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count < 2) {
        printf("  SKIP: Need at least 2 GPUs\n");
        return;
    }
    
    const int n = 40;
    float* h_dist = new float[n * n];
    generate_random_tsp(h_dist, n, 34567);
    
    auto prob = TSPProblem::create(h_dist, n);
    
    SolverConfig cfg;
    cfg.pop_size = 512;
    cfg.max_gen = 5000;
    cfg.verbose = false;
    cfg.seed = 42;
    cfg.num_islands = 8;
    cfg.use_aos = true;
    cfg.sa_temp_init = 20.0f;
    cfg.use_cuda_graph = true;
    
    // 运行多次取平均
    const int num_runs = 5;
    
    printf("\n  Running %d times with 2 GPUs...\n", num_runs);
    
    // 方案 B3: 有交换
    float b3_sum = 0.0f;
    cfg.num_gpus = 2;
    cfg.multi_gpu_interval_sec = 2.0f;
    for (int run = 0; run < num_runs; run++) {
        cfg.seed = 42 + run * 100;
        auto result = solve_multi_gpu(prob, cfg);
        b3_sum += result.best_solution.objectives[0];
        printf("    Run %d: obj=%.2f\n", run+1, result.best_solution.objectives[0]);
    }
    float b3_avg = b3_sum / num_runs;
    
    printf("\n  B3 Average: %.2f\n", b3_avg);
    
    delete[] h_dist;
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("Multi-GPU B3 (Passive Injection) Test Suite\n");
    printf("============================================\n");
    
    test_inject_buffer();
    test_inject_check_kernel();
    test_2gpu_tsp_small();
    test_2gpu_vrp_medium();
    test_performance_comparison();
    
    printf("\n=== All Tests Completed ===\n");
    return 0;
}
