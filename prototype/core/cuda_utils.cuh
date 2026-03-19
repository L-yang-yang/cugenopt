/**
 * cuda_utils.cuh - CUDA 工具集
 * 
 * 职责：错误检查、设备信息、随机数工具
 * 规则：所有 CUDA API 调用都必须用 CUDA_CHECK 包裹
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <curand_kernel.h>

// ============================================================
// 错误检查
// ============================================================

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// kernel launch 后检查（捕获异步错误）
#define CUDA_CHECK_LAST() do {                                      \
    cudaError_t err = cudaGetLastError();                            \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n",        \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// ============================================================
// 设备信息
// ============================================================

inline void print_device_info() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("GPU: %s\n", prop.name);
    printf("  SM count:       %d\n", prop.multiProcessorCount);
    printf("  Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Shared mem/blk: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Global mem:     %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Compute cap:    %d.%d\n", prop.major, prop.minor);
}

// ============================================================
// 随机数工具 (Device 端)
// ============================================================

// 初始化 curand 状态，每个线程一个
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Device 端：生成 [0, bound) 的随机整数
__device__ inline int rand_int(curandState* state, int bound) {
    return curand(state) % bound;
}

// Device 端：Fisher-Yates shuffle，对 arr[0..n-1] 做随机排列
__device__ inline void shuffle(int* arr, int n, curandState* state) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand_int(state, i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

// ============================================================
// Kernel 启动参数计算
// ============================================================

inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

// 计算合适的 block 数量
inline int calc_grid_size(int n, int block_size = 256) {
    return div_ceil(n, block_size);
}
