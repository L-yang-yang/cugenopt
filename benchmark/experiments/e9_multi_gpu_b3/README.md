# E9: Multi-GPU B3 方案验证

## 实验目的

验证 Multi-GPU v5.0 方案 B3（被动注入）在运行期间进行解交换的有效性，对比简化版（独立运行 + 最终比较）。

## 实验设计

### 对比方案

1. **简化版（Baseline）**: 在单 GPU 上运行多次独立 `solve()`，每次使用不同种子，最后选择最优解
2. **B3 保守策略**: `interval=3s`, `MultiGpuInjectMode::OneIsland` 或 `HalfIslands`
3. **B3 激进策略**: `interval=1s`, `MultiGpuInjectMode::AllIslands`

### 测试问题

| 问题 | 规模 | 说明 |
|------|------|------|
| TSP | n=50 | 小规模基准测试 |
| TSP | n=64 | 最大支持规模（受 `Solution<1,64>` 限制） |
| VRP | n=40 | 中等规模约束问题 |
| VRP | n=50 | 较大规模约束问题（遇到内存错误） |

### 配置参数

```cpp
SolverConfig cfg;
cfg.pop_size = 1024;
cfg.max_gen = 10000;
cfg.num_islands = 16;
cfg.use_aos = true;
cfg.sa_temp_init = 50.0f;
cfg.use_cuda_graph = true;
cfg.num_gpus = 2;  // B3 方案
```

### 运行环境

- **GPU**: 2×V100S (16GB)
- **CUDA**: 12.8
- **运行次数**: 每个配置 5-10 次取平均

## 实验结果

### 小规模问题（TSP n=50, VRP n=40）

| 问题 | 简化版 | B3 保守 | B3 激进 | 改进（保守） | 改进（激进） |
|------|--------|---------|---------|-------------|-------------|
| TSP n=50 | 712.76 | 712.83 | 712.78 | **-0.01%** | **-0.00%** |
| VRP n=40 | 786.00 | 786.00 | 786.53 | **0.00%** | **-0.07%** |

**运行次数**: 10 次平均

### 大规模问题（TSP n=64）

| 问题 | 简化版 | B3 激进 | 改进 |
|------|--------|---------|------|
| TSP n=64 | 825.37 | 825.27 | **+0.01%** |

**运行次数**: 8 次平均

### 详细数据（TSP n=64, 8 runs）

#### 简化版
```
Run 1: 830.20
Run 2: 824.20
Run 3: 825.40
Run 4: 825.00
Run 5: 823.60
Run 6: 824.40
Run 7: 823.10
Run 8: 827.10
平均: 825.37
```

#### B3 激进（interval=1s, AllIslands）
```
Run 1: 830.80
Run 2: 828.80
Run 3: 821.00
Run 4: 824.10
Run 5: 823.20
Run 6: 825.10
Run 7: 822.00
Run 8: 827.20
平均: 825.27
```

## 结论

### 主要发现

1. **B3 方案未带来显著收益**: 在所有测试规模上，B3（运行期间解交换）相比简化版（独立运行）的改进均在 ±0.1% 范围内，属于统计噪声
2. **问题规模影响不大**: 从小规模（n=50）到大规模（n=64），B3 的相对表现没有明显变化
3. **注入策略影响微弱**: 保守策略（3s, OneIsland）和激进策略（1s, AllIslands）的效果差异不明显

### 技术分析

#### 为什么 B3 没有效果？

1. **搜索空间特性**: 元启发式算法的搜索轨迹高度依赖初始解和随机种子，不同 GPU 的搜索轨迹本质上是相互独立的
2. **解的多样性不足**: 不同 GPU 找到的最优解往往处于相似的局部最优区域，注入到其他 GPU 后无法带来新的搜索方向
3. **注入时机问题**: 在搜索中期注入外部解可能破坏已有的搜索动量，反而降低收敛效率
4. **岛屿模型已足够**: 单 GPU 内部的 16 个岛屿已经提供了足够的种群多样性

#### 与行业实践一致

- **cuOpt**: NVIDIA 官方组合优化求解器不支持多 GPU
- **OR-Tools**: Google 的求解器不支持多 GPU
- **Gurobi/CPLEX**: 商业 MIP 求解器的多 GPU 支持仅限于特定算法（如 Barrier）

这些商业求解器的选择说明：**对于组合优化问题，多 GPU 的投入产出比很低**。

### 规模限制

当前测试受到以下限制：

1. **编码维度**: `TSPProblem` 的 `D2=64` 限制了最大问题规模为 n=64
2. **VRP 内存错误**: VRP n≥50 时出现 `illegal memory access`，可能是 VRP 编码的内存布局问题
3. **GPU 资源**: 仅有 2×V100S 可用，无法测试 4 GPU 的效果

**用户观点**: "本质还是我们的规模太小了，GPU 解决的 TSP 应该是千级别的"——这是合理的观察。真正需要多 GPU 协同的问题规模应该在 n>1000，但当前框架的编码限制（固定维度数组）无法支持。

## 下一步建议

### 短期（暂缓）

- **标记为探索性功能**: 将 B3 方案标记为"技术可行但效果不明显"，不作为主要卖点
- **保留代码**: B3 的实现（`InjectBuffer`, `inject_check_kernel`, `coordinator_thread`）技术上是正确的，可以保留作为框架能力展示

### 长期（如需要）

- **突破编码限制**: 实现动态维度编码（如 `std::vector` 或 GPU 端动态分配），支持 n>1000 的超大规模问题
- **重新评估**: 在千级规模上重新测试 B3 方案，此时多 GPU 的价值可能显现
- **探索其他多 GPU 模式**: 如问题分解（Domain Decomposition）而非解交换

## 文件清单

### 实验代码（远程 gpu2v100）

- `~/cugenopt_b3/test_b3_benchmark.cu`: 初始 B3 vs 1-GPU 对比（TSP n=50, VRP n=40）
- `~/cugenopt_b3/test_b3_vs_simplified.cu`: B3 vs 简化版直接对比（TSP n=50, VRP n=40）
- `~/cugenopt_b3/test_b3_aggressive.cu`: 激进策略测试（3 种策略对比）
- `~/cugenopt_b3/test_b3_final.cu`: 大规模测试（TSP n=64, VRP n=50）

### 核心实现

- `prototype/core/types.cuh`: `InjectBuffer` 结构定义
- `prototype/core/solver.cuh`: `inject_check_kernel` 实现
- `prototype/core/multi_gpu_solver.cuh`: `coordinator_thread` 和 `solve_multi_gpu` 实现

### 设计文档

- `MULTI_GPU_EXCHANGE_DESIGN.md`: 完整的方案设计和技术分析
- `MULTI_GPU_INDUSTRY_PATTERNS.md`: 行业多 GPU 模式调研
- `MULTI_GPU_COUPLING_ANALYSIS.md`: 耦合度分析

---

**实验日期**: 2026-03-05  
**最后更新**: 2026-03-05
