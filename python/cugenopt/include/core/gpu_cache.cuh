/**
 * gpu_cache.cuh - GPU 全局内存哈希表（通用缓存组件）
 * 
 * 设计：
 *   - 开放寻址，固定容量（power of 2），线性探测
 *   - key = uint64_t（由 Problem 自行计算 hash）
 *   - value = float（单个指标值）
 *   - 无锁：允许 race condition（缓存语义，偶尔脏读可接受）
 *   - 自带命中/未命中原子计数器
 * 
 * 用法：
 *   GpuCache cache = GpuCache::allocate(65536);   // host
 *   // ... pass cache as Problem member to kernels ...
 *   cache.print_stats();                           // host
 *   cache.destroy();                               // host
 * 
 * 参考：scute 项目 LRUCache（key = metric_type + content_hash）
 */

#pragma once
#include "cuda_utils.cuh"
#include <cstdint>

// ============================================================
// 常量
// ============================================================

static constexpr uint64_t CACHE_EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL;
static constexpr int CACHE_MAX_PROBE = 8;   // 最大线性探测步数

// ============================================================
// GpuCache 结构体（POD，可安全拷贝到 kernel）
// ============================================================

struct GpuCache {
    uint64_t* keys;             // GPU 全局内存
    float*    values;           // GPU 全局内存
    unsigned int* d_hits;       // 原子计数器（GPU）
    unsigned int* d_misses;     // 原子计数器（GPU）
    int capacity;               // 必须是 2 的幂
    int mask;                   // = capacity - 1
    
    // ---- Host 操作 ----
    
    static GpuCache allocate(int cap = 65536) {
        GpuCache c;
        c.capacity = cap;
        c.mask = cap - 1;
        CUDA_CHECK(cudaMalloc(&c.keys,     sizeof(uint64_t) * cap));
        CUDA_CHECK(cudaMalloc(&c.values,   sizeof(float) * cap));
        CUDA_CHECK(cudaMalloc(&c.d_hits,   sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&c.d_misses, sizeof(unsigned int)));
        c.clear();
        return c;
    }
    
    static GpuCache disabled() {
        GpuCache c;
        c.keys = nullptr;  c.values = nullptr;
        c.d_hits = nullptr; c.d_misses = nullptr;
        c.capacity = 0;  c.mask = 0;
        return c;
    }
    
    bool is_enabled() const { return keys != nullptr; }
    
    void clear() {
        CUDA_CHECK(cudaMemset(keys, 0xFF, sizeof(uint64_t) * capacity));
        CUDA_CHECK(cudaMemset(d_hits,   0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_misses, 0, sizeof(unsigned int)));
    }
    
    void destroy() {
        if (keys)     cudaFree(keys);
        if (values)   cudaFree(values);
        if (d_hits)   cudaFree(d_hits);
        if (d_misses) cudaFree(d_misses);
        keys = nullptr; values = nullptr;
        d_hits = nullptr; d_misses = nullptr;
    }
    
    void print_stats() const {
        if (!keys) { printf("  Cache: disabled\n"); return; }
        unsigned int h = 0, m = 0;
        CUDA_CHECK(cudaMemcpy(&h, d_hits,   sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&m, d_misses, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        unsigned int total = h + m;
        float rate = total > 0 ? (float)h / total * 100.0f : 0.0f;
        printf("  Cache: %u lookups | %u hits + %u misses | hit rate = %.1f%%\n",
               total, h, m, rate);
        printf("  Cache: capacity = %d entries (%.1f KB)\n",
               capacity, capacity * (sizeof(uint64_t) + sizeof(float)) / 1024.0f);
    }
};

// ============================================================
// Device 函数：哈希 / 查找 / 插入
// ============================================================

/// FNV-1a 哈希：对一段有序 int 序列（如路线中的客户 ID）
__device__ inline uint64_t route_hash(const int* data, int len) {
    uint64_t h = 14695981039346656037ULL;   // FNV offset basis
    for (int i = 0; i < len; i++) {
        h ^= (uint64_t)(unsigned int)data[i];
        h *= 1099511628211ULL;               // FNV prime
    }
    return (h == CACHE_EMPTY_KEY) ? h - 1 : h;  // 避免与哨兵值碰撞
}

/// 查找：命中返回 true + 写入 out
__device__ inline bool cache_lookup(const GpuCache& c, uint64_t key, float& out) {
    int slot = (int)(key & (uint64_t)c.mask);
    for (int p = 0; p < CACHE_MAX_PROBE; p++) {
        int idx = (slot + p) & c.mask;
        uint64_t k = c.keys[idx];
        if (k == key) {
            out = c.values[idx];
            return true;
        }
        if (k == CACHE_EMPTY_KEY) return false;  // 空槽 → 一定不存在
    }
    return false;   // 探测用尽
}

/// 插入：写入 key-value，同 key 覆盖，探测满则驱逐首槽
__device__ inline void cache_insert(const GpuCache& c, uint64_t key, float value) {
    int slot = (int)(key & (uint64_t)c.mask);
    for (int p = 0; p < CACHE_MAX_PROBE; p++) {
        int idx = (slot + p) & c.mask;
        uint64_t k = c.keys[idx];
        if (k == CACHE_EMPTY_KEY || k == key) {
            c.keys[idx]   = key;
            c.values[idx] = value;
            return;
        }
    }
    // 探测满：驱逐首槽
    int idx = slot & c.mask;
    c.keys[idx]   = key;
    c.values[idx] = value;
}
