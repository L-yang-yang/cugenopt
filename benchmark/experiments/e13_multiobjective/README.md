# E13: 多目标优化验证实验

## 实验目标

验证 cuGenOpt 框架的两种多目标比较模式：
1. **Weighted（加权求和）** - 目标可权衡
2. **Lexicographic（字典法）** - 目标有严格优先级

## 实验内容

### 主实验（单 GPU）

1. **双目标 VRP (A-n32-k5)**
   - 目标：最小化总距离 + 最小化车辆数
   - Weighted 模式：3 组权重配置 `[0.9,0.1]`, `[0.7,0.3]`, `[0.5,0.5]`
   - Lexicographic 模式：3 组配置（不同优先级和容差）

2. **三目标 VRP (A-n32-k5)**
   - 目标：最小化总距离 + 最小化车辆数 + 最小化最大路径长度
   - Weighted 模式：1 组权重配置 `[0.6,0.2,0.2]`
   - Lexicographic 模式：2 组配置（不同优先级顺序）

3. **双目标 Knapsack (knapPI_1_100)**
   - 目标：最大化价值 + 最小化重量
   - Weighted 模式：1 组权重配置 `[0.8,0.2]`
   - Lexicographic 模式：1 组配置（优先级 [价值, 重量]）

### 附加验证（多 GPU）

- 双目标 VRP (A-n32-k5)
- Weighted 模式：`[0.7,0.3]`
- Lexicographic 模式：优先级 [距离, 车辆数]
- 2×T4, 60 秒, 单次运行

## 编译和运行

### 在 gpu2v100 上编译

```bash
cd /path/to/generic_solver/benchmark/experiments/e13_multiobjective
make
```

### 运行实验

```bash
./e13_multiobjective > e13_results.txt 2>&1
```

## 文件说明

- `bi_objective_vrp.cuh` - 双目标 VRP Problem 定义
- `tri_objective_vrp.cuh` - 三目标 VRP Problem 定义
- `bi_objective_knapsack.cuh` - 双目标 Knapsack Problem 定义
- `gpu.cu` - 主实验程序
- `Makefile` - 编译配置
- `DESIGN.md` - 详细实验设计文档

## 预期输出

每个配置运行 5 次（seeds: 42, 123, 456, 789, 2024），输出格式：

```
[BiVRP] W_90_10 (mode=Weighted, multi_gpu=NO)
  Run 1 (seed=42): obj0=850.23 obj1=6.00 penalty=0.00 time=60.0s gen=12345
  Run 2 (seed=123): obj0=845.67 obj1=6.00 penalty=0.00 time=60.0s gen=12456
  ...
```

## 数据分析

实验完成后，运行数据分析脚本生成报告：

```bash
python3 analyze_results.py e13_results.txt
```

将生成 `E13_REPORT.md` 包含：
- Weighted 不同权重下的解质量对比表
- Lexicographic 不同容差下的解质量对比表
- 多 GPU 验证结果
