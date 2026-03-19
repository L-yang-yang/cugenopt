# 延迟归一化测试

## 目的

验证延迟归一化（Lazy Normalization）机制的正确性和性能。

## 核心修改

### 1. SeqRegistry 结构

```cpp
struct SeqRegistry {
    int   ids[MAX_SEQ];
    int   count;
    float weights[MAX_SEQ];   // 未归一化
    float weights_sum;        // 缓存权重和 ⭐ 新增
    float max_w[MAX_SEQ];
    SeqCategory categories[MAX_SEQ];
};
```

### 2. 轮盘赌选择

```cpp
// 原来：r ∈ [0, 1)，要求权重归一化
float r = curand_uniform(rng);

// 现在：r ∈ [0, weights_sum)，不要求权重归一化
float r = curand_uniform(rng) * reg.weights_sum;
```

### 3. AOS 更新

```cpp
// 原来：EMA 更新 → 归一化 → FLOOR/CAP → 再次归一化
// 现在：EMA 更新 → FLOOR/CAP → 更新 weights_sum（不归一化）
```

## 编译和运行

```bash
# 在 gpu1v100 上编译
make

# 运行测试
./test_lazy_norm
```

## 预期输出

```
=== 延迟归一化测试 ===

配置:
  pop_size = 32
  max_gen = 100
  aos_weight_floor = 0.050
  aos_weight_cap = 0.350
  延迟归一化: 启用

开始求解...

  [AOS batch g=10] usage: ... | w: 0.xxx 0.xxx ... | sum=0.xxx | K: ...
  [AOS batch g=20] usage: ... | w: 0.xxx 0.xxx ... | sum=0.xxx | K: ...
  ...

=== 求解完成 ===
最优解: xxx.xx
代数: 100
时间: xxx.xx ms

✅ 延迟归一化测试通过！
```

## 验证要点

1. **权重和可能 ≠ 1.0**：`sum=0.xxx`（正常）
2. **权重在边界内**：所有 `w[i] ∈ [0.05, 0.35]`
3. **求解正常完成**：无崩溃、无异常
4. **结果合理**：找到可行解
