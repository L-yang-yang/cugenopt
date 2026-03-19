/**
 * relation_matrix.cuh - G/O 关系矩阵管理
 *
 * G[i][j]: 分组倾向（元素 i 和 j 应在同一行的倾向，对称）
 * O[i][j]: 排序倾向（元素 i 应排在 j 前面的倾向，不对称）
 *
 * 更新来源：历史最优解统计
 *   每当 host 端获取到当前 best 解，扫描所有元素对关系：
 *     - 同行 → G[i][j] 增强
 *     - i 在 j 前 → O[i][j] 增强
 *   使用 EMA 衰减：M[i][j] = α * M[i][j] + (1-α) * signal
 *
 * 生命周期：
 *   1. relation_matrix_create(N)  — 分配 host/device 内存，初始化为 0
 *   2. relation_matrix_update(rm, sol, dim1) — 从一个解更新 G/O（host 端）
 *   3. relation_matrix_upload(rm) — 上传 h_G/h_O 到 d_G/d_O
 *   4. relation_matrix_destroy(rm) — 释放内存
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include <cstring>

// ============================================================
// 创建 / 销毁
// ============================================================

inline RelationMatrix relation_matrix_create(int N, float decay = 0.95f) {
    RelationMatrix rm;
    rm.N = N;
    rm.decay = decay;
    rm.update_count = 0;
    
    size_t bytes = (size_t)N * N * sizeof(float);
    
    rm.h_G = new float[N * N];
    rm.h_O = new float[N * N];
    memset(rm.h_G, 0, bytes);
    memset(rm.h_O, 0, bytes);
    
    CUDA_CHECK(cudaMalloc(&rm.d_G, bytes));
    CUDA_CHECK(cudaMalloc(&rm.d_O, bytes));
    CUDA_CHECK(cudaMemset(rm.d_G, 0, bytes));
    CUDA_CHECK(cudaMemset(rm.d_O, 0, bytes));
    
    return rm;
}

inline void relation_matrix_destroy(RelationMatrix& rm) {
    delete[] rm.h_G;
    delete[] rm.h_O;
    CUDA_CHECK(cudaFree(rm.d_G));
    CUDA_CHECK(cudaFree(rm.d_O));
    rm.h_G = rm.h_O = nullptr;
    rm.d_G = rm.d_O = nullptr;
    rm.N = 0;
}

// ============================================================
// 从一个解更新 G/O（host 端）
// ============================================================
// sol: 当前最优解（已下载到 host）
// dim1: 实际使用的行数
//
// 逻辑：
//   对 sol 中每对元素 (val_a, val_b)：
//     如果在同一行 → G[val_a][val_b] 增强
//     如果 val_a 在 val_b 前面 → O[val_a][val_b] 增强
//
// 注意：元素值 val 必须在 [0, N) 范围内才有意义
//       对于 partition 编码（VRP），元素值就是客户编号
//       对于单行排列（TSP），元素值就是城市编号

template<typename Sol>
void relation_matrix_update(RelationMatrix& rm, const Sol& sol, int dim1) {
    int N = rm.N;
    float alpha = rm.decay;
    float signal_strength = 1.0f;
    
    // 衰减所有现有值
    for (int i = 0; i < N * N; i++) {
        rm.h_G[i] *= alpha;
        rm.h_O[i] *= alpha;
    }
    
    // 扫描解中的元素对关系
    for (int r = 0; r < dim1; r++) {
        int sz = sol.dim2_sizes[r];
        for (int c1 = 0; c1 < sz; c1++) {
            int val_a = sol.data[r][c1];
            if (val_a < 0 || val_a >= N) continue;
            
            for (int c2 = c1 + 1; c2 < sz; c2++) {
                int val_b = sol.data[r][c2];
                if (val_b < 0 || val_b >= N) continue;
                
                // 同行 → G 增强（对称）
                rm.h_G[val_a * N + val_b] += (1.0f - alpha) * signal_strength;
                rm.h_G[val_b * N + val_a] += (1.0f - alpha) * signal_strength;
                
                // val_a 在 val_b 前 → O[val_a][val_b] 增强
                rm.h_O[val_a * N + val_b] += (1.0f - alpha) * signal_strength;
            }
        }
    }
    
    // 裁剪到 [0, 1]
    for (int i = 0; i < N * N; i++) {
        if (rm.h_G[i] > 1.0f) rm.h_G[i] = 1.0f;
        if (rm.h_O[i] > 1.0f) rm.h_O[i] = 1.0f;
    }
    
    rm.update_count++;
}

// ============================================================
// 上传到 GPU
// ============================================================

inline void relation_matrix_upload(const RelationMatrix& rm) {
    size_t bytes = (size_t)rm.N * rm.N * sizeof(float);
    CUDA_CHECK(cudaMemcpy(rm.d_G, rm.h_G, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rm.d_O, rm.h_O, bytes, cudaMemcpyHostToDevice));
}
