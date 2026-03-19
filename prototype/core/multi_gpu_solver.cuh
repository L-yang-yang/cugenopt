/**
 * multi_gpu_solver.cuh - 多 GPU 协同求解
 * 
 * v5.0 方案 B3: 被动注入 + GPU 无感知
 *   - 每块 GPU 独立运行 solve()，各自用不同 seed
 *   - 每个 GPU 有一个 InjectBuffer（设备端）
 *   - CPU 协调线程定期（每 N 秒）收集各 GPU 的 best，异步写入其他 GPU 的 InjectBuffer
 *   - GPU 在 migrate_kernel 后检查 InjectBuffer，如果有新解则注入
 *   - 完全解耦：GPU 无需暂停，CPU 异步写入，通过 CUDA Stream 同步保证安全
 */

#pragma once
#include "solver.cuh"
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <chrono>

// ============================================================
// MultiGpuContext — 每个 GPU 的上下文
// ============================================================

template<typename Problem>
struct MultiGpuContext {
    using Sol = typename Problem::Sol;
    
    int gpu_id;                      // GPU 设备 ID
    Problem* problem;                // Problem 实例（设备指针指向该 GPU）
    SolverConfig config;             // 求解器配置（独立 seed）
    
    Sol best_solution;               // 当前最优解（host 端）
    std::mutex best_mutex;           // 保护 best_solution 的互斥锁
    
    InjectBuffer<Sol>* d_inject_buf; // Device 端注入缓冲区（在该 GPU 上分配）
    Sol* d_global_best;              // Device 端全局最优解指针（由 solve() 导出）
    
    std::atomic<bool> stop_flag;     // 停止标志
    std::atomic<bool> running;       // 运行状态标志（用于协调线程判断）
    
    MultiGpuContext(int id) : gpu_id(id), problem(nullptr), d_inject_buf(nullptr), 
                               d_global_best(nullptr), stop_flag(false), running(false) {
        best_solution = Sol{};
        best_solution.penalty = 1e30f;
        for (int i = 0; i < MAX_OBJ; i++) best_solution.objectives[i] = 1e30f;
    }
};

// ============================================================
// GPU Worker 线程函数（方案 B3）
// ============================================================

template<typename Problem>
void gpu_worker(MultiGpuContext<Problem>* ctx) {
    using Sol = typename Problem::Sol;
    
    // 设置当前线程使用的 GPU
    CUDA_CHECK(cudaSetDevice(ctx->gpu_id));
    
    // 标记开始运行
    ctx->running.store(true);
    
    // 运行 solve（传入 inject_buf 和 d_global_best_out）
    SolveResult<Sol> result = solve(*ctx->problem, ctx->config, 
                                     nullptr, 0, nullptr, ctx->d_inject_buf, &ctx->d_global_best);
    
    // 标记运行结束
    ctx->running.store(false);
    
    // 更新最优解
    {
        std::lock_guard<std::mutex> lock(ctx->best_mutex);
        ctx->best_solution = result.best_solution;
    }
    
    // 标记完成
    ctx->stop_flag.store(true);
}

// ============================================================
// 协调线程函数（方案 B3）
// ============================================================
// 定期从各 GPU 的 d_global_best 读取当前 best，计算 global_best，注入到其他 GPU
//
// 关键设计：
// 1. 直接从各 GPU 的 d_global_best 读取（由 solve() 导出）
// 2. 要求启用 SA（否则无 d_global_best）
// 3. 轻量侵入：solve() 只需导出一个指针，对单 GPU 无影响

template<typename Problem>
void coordinator_thread(std::vector<MultiGpuContext<Problem>*>& contexts,
                        float interval_sec, bool verbose) {
    using Sol = typename Problem::Sol;
    ObjConfig oc = contexts[0]->problem->obj_config();
    
    auto interval_ms = std::chrono::milliseconds(static_cast<int>(interval_sec * 1000));
    int round = 0;
    
    // 等待所有 GPU 的 d_global_best 就绪
    bool all_ready = false;
    while (!all_ready) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        all_ready = true;
        for (auto* ctx : contexts) {
            if (ctx->d_global_best == nullptr && ctx->running.load()) {
                all_ready = false;
                break;
            }
        }
    }
    
    while (true) {
        // 等待指定时间间隔
        std::this_thread::sleep_for(interval_ms);
        
        // 检查是否所有 GPU 都已停止
        bool all_stopped = true;
        for (auto* ctx : contexts) {
            if (ctx->running.load()) {
                all_stopped = false;
                break;
            }
        }
        if (all_stopped) break;
        
        round++;
        
        // 收集各 GPU 的当前最优解（从 d_global_best 读取）
        Sol global_best;
        global_best.penalty = 1e30f;
        global_best.objectives[0] = 1e30f;
        int best_gpu = -1;
        
        for (int i = 0; i < (int)contexts.size(); i++) {
            if (!contexts[i]->running.load()) continue;  // 已停止的 GPU 跳过
            if (contexts[i]->d_global_best == nullptr) continue;  // 未就绪跳过
            
            // 从该 GPU 的 d_global_best 读取
            Sol gpu_best;
            cudaSetDevice(contexts[i]->gpu_id);
            cudaMemcpy(&gpu_best, contexts[i]->d_global_best, sizeof(Sol), cudaMemcpyDeviceToHost);
            
            if (best_gpu == -1 || is_better(gpu_best, global_best, oc)) {
                global_best = gpu_best;
                best_gpu = i;
            }
        }
        
        if (best_gpu == -1) continue;  // 所有 GPU 都已停止或未就绪
        
        if (verbose) {
            printf("  [Coordinator Round %d] Global best from GPU %d: obj=%.2f, penalty=%.2f\n",
                   round, best_gpu, global_best.objectives[0], global_best.penalty);
        }
        
        // 将 global_best 注入到其他 GPU（除了 best_gpu 自己）
        for (int i = 0; i < (int)contexts.size(); i++) {
            if (i == best_gpu) continue;  // 不注入到自己
            if (!contexts[i]->running.load()) continue;  // 已停止的 GPU 不注入
            
            // 读取 InjectBuffer 结构（从 device 到 host）
            InjectBuffer<Sol> buf;
            cudaMemcpy(&buf, contexts[i]->d_inject_buf, sizeof(InjectBuffer<Sol>), cudaMemcpyDeviceToHost);
            
            // 同步写入（会自动切换设备）
            buf.write_sync(global_best, contexts[i]->gpu_id);
        }
    }
    
    if (verbose) {
        printf("  [Coordinator] All GPUs stopped, coordinator exiting.\n");
    }
}

// ============================================================
// 多 GPU 协同求解主函数（方案 B3）
// ============================================================

template<typename Problem>
SolveResult<typename Problem::Sol> solve_multi_gpu(Problem& prob, const SolverConfig& cfg) {
    using Sol = typename Problem::Sol;
    
    if (cfg.num_gpus <= 1) {
        // 单 GPU 模式，直接调用普通 solve
        return solve(prob, cfg);
    }
    
    // 检查可用 GPU 数量
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int actual_gpus = std::min(cfg.num_gpus, device_count);
    
    if (cfg.verbose) {
        printf("  [Multi-GPU B3] Using %d GPUs (requested %d, available %d)\n",
               actual_gpus, cfg.num_gpus, device_count);
        printf("  [Multi-GPU B3] Exchange interval: %.1fs, inject mode: %s\n",
               cfg.multi_gpu_interval_sec,
               cfg.multi_gpu_inject_mode == MultiGpuInjectMode::OneIsland ? "OneIsland" :
               cfg.multi_gpu_inject_mode == MultiGpuInjectMode::HalfIslands ? "HalfIslands" : "AllIslands");
    }
    
    // 创建各 GPU 的上下文
    std::vector<MultiGpuContext<Problem>*> contexts;
    for (int i = 0; i < actual_gpus; i++) {
        auto* ctx = new MultiGpuContext<Problem>(i);
        ctx->config = cfg;
        ctx->config.seed = cfg.seed + i * 1000;  // 每个 GPU 用不同 seed
        ctx->config.num_gpus = 1;  // 单 GPU 模式运行
        
        // 克隆 Problem 到该 GPU
        ctx->problem = prob.clone_to_device(i);
        if (ctx->problem == nullptr) {
            fprintf(stderr, "Error: Failed to clone problem to GPU %d\n", i);
            for (auto* c : contexts) {
                if (c->problem) delete c->problem;
                delete c;
            }
            return SolveResult<Sol>{};
        }
        
        // 分配 InjectBuffer（在该 GPU 上）
        InjectBuffer<Sol> buf = InjectBuffer<Sol>::allocate(i);
        
        // 将 InjectBuffer 拷贝到 device 端（传给 kernel）
        InjectBuffer<Sol>* d_buf;
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_buf, sizeof(InjectBuffer<Sol>)));
        CUDA_CHECK(cudaMemcpy(d_buf, &buf, sizeof(InjectBuffer<Sol>), cudaMemcpyHostToDevice));
        ctx->d_inject_buf = d_buf;
        
        contexts.push_back(ctx);
    }
    
    // 启动 worker 线程
    std::vector<std::thread> workers;
    for (auto* ctx : contexts) {
        workers.emplace_back(gpu_worker<Problem>, ctx);
    }
    
    // 启动协调线程（定期注入 global_best）
    std::thread coordinator(coordinator_thread<Problem>, std::ref(contexts),
                            cfg.multi_gpu_interval_sec, cfg.verbose);
    
    // 等待所有 worker 完成
    for (auto& w : workers) w.join();
    
    // 等待协调线程完成
    coordinator.join();
    
    // 收集最终结果
    Sol final_best = contexts[0]->best_solution;
    ObjConfig oc = prob.obj_config();
    for (int i = 1; i < (int)contexts.size(); i++) {
        if (is_better(contexts[i]->best_solution, final_best, oc)) {
            final_best = contexts[i]->best_solution;
        }
    }
    
    // 清理
    for (auto* ctx : contexts) {
        // 读取 InjectBuffer 的内容（用于释放）
        InjectBuffer<Sol> buf;
        CUDA_CHECK(cudaSetDevice(ctx->gpu_id));
        CUDA_CHECK(cudaMemcpy(&buf, ctx->d_inject_buf, sizeof(InjectBuffer<Sol>), cudaMemcpyDeviceToHost));
        buf.destroy();
        CUDA_CHECK(cudaFree(ctx->d_inject_buf));
        
        if (ctx->problem) delete ctx->problem;
        delete ctx;
    }
    
    // 构造返回结果
    SolveResult<Sol> result;
    result.best_solution = final_best;
    result.stop_reason = StopReason::MaxGen;
    
    return result;
}
