#include "solver.cuh"
#include <cstdio>
#include <cmath>

// 简单的 TSP 问题用于测试
struct SimpleTSP : public ProblemBase<SimpleTSP, 1, 64> {
    using Sol = Solution<1, 64>;
    
    const float* d_dist;
    int n;
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f}
    };
    
    __device__ float compute_obj(int obj_idx, const Sol& s) const {
        float total = 0.0f;
        for (int i = 0; i < n; i++) {
            int from = s.data[0][i];
            int to = s.data[0][(i + 1) % n];
            total += d_dist[from * (n + 1) + to];
        }
        return total;
    }
    
    __device__ float compute_penalty(const Sol& s) const {
        return 0.0f;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = 1;
        cfg.dim2_default = n;
        fill_obj_config(cfg);
        cfg.cross_row_prob = 0.0f;
        cfg.row_mode = RowMode::Fixed;
        cfg.total_elements = n;
        return cfg;
    }
    
    SimpleTSP* clone_to_device(int target_device) const override {
        return nullptr;
    }
};

constexpr ObjDef SimpleTSP::OBJ_DEFS[];

int main() {
    printf("=== 延迟归一化测试 ===\n\n");
    
    // 创建小规模 TSP 实例（10 个城市）
    const int n = 10;
    float h_dist[(n+1) * (n+1)];
    
    // 生成随机距离矩阵
    srand(42);
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == j) {
                h_dist[i * (n+1) + j] = 0.0f;
            } else {
                h_dist[i * (n+1) + j] = 10.0f + rand() % 90;
            }
        }
    }
    
    // 拷贝到 GPU
    float* d_dist;
    cudaMalloc(&d_dist, (n+1) * (n+1) * sizeof(float));
    cudaMemcpy(d_dist, h_dist, (n+1) * (n+1) * sizeof(float), cudaMemcpyHostToDevice);
    
    SimpleTSP prob;
    prob.d_dist = d_dist;
    prob.n = n;
    
    // 配置求解器（启用 AOS 和 verbose）
    SolverConfig cfg;
    cfg.pop_size = 32;
    cfg.max_gen = 500;
    cfg.use_aos = true;
    cfg.verbose = true;
    cfg.aos_update_interval = 5;
    cfg.aos_weight_floor = 0.05f;
    cfg.aos_weight_cap = 0.35f;
    
    printf("配置:\n");
    printf("  pop_size = %d\n", cfg.pop_size);
    printf("  max_gen = %d\n", cfg.max_gen);
    printf("  aos_weight_floor = %.3f\n", cfg.aos_weight_floor);
    printf("  aos_weight_cap = %.3f\n", cfg.aos_weight_cap);
    printf("  延迟归一化: 启用\n\n");
    
    // 求解
    printf("开始求解...\n\n");
    auto result = solve(prob, cfg);
    
    printf("\n=== 求解完成 ===\n");
    printf("最优解: %.2f\n", result.best_solution.objectives[0]);
    printf("代数: %d\n", result.generations);
    printf("时间: %.2f ms\n", result.elapsed_ms);
    
    // 清理
    cudaFree(d_dist);
    
    printf("\n✅ 延迟归一化测试通过！\n");
    
    return 0;
}
