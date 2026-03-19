# E10: 大规模问题实验

## 实验目的

验证 cuGenOpt 在大规模问题（n>100）上的性能表现，以及多 GPU 简化版的实际收益。

## 实验设计

### 测试规模

**TSP**:
- n = 100, 200, 300, 400, 500

**VRP**:
- n = 50, 100, 150, 200
- 车辆数动态调整（n/20 + 1）
- 容量固定为 150

### 对比维度

1. **单 GPU vs 多 GPU**（简化版）
2. **不同规模下的性能表现**
3. **多 GPU 的收益曲线**

### 配置参数

```cpp
SolverConfig cfg;
cfg.pop_size = 0;           // 自适应（L2 cache感知）
cfg.max_gen = 10000;
cfg.num_islands = 16;
cfg.use_aos = true;
cfg.sa_temp_init = 50.0f;
cfg.use_cuda_graph = true;
```

### 运行次数

每个配置运行 5 次，取平均值。

## 文件说明

- `large_tsp_problem.cuh`: 支持最多 512 个城市的 TSP 问题定义
- `large_vrp_problem.cuh`: 支持最多 256 个客户、16 辆车的 VRP 问题定义
- `gpu.cu`: 主实验代码

## 编译和运行

```bash
# 在远程服务器上
cd ~/cugenopt_e10

# 编译
nvcc -arch=sm_70 -O2 -std=c++17 --extended-lambda \
     -I ../../../prototype/core \
     -I ../../../prototype/problems \
     -I . \
     -o e10_test gpu.cu

# 运行
./e10_test > e10_output.txt 2>&1
```

## 预期结果

1. **单 GPU 性能**：
   - 小规模（n≤100）：gap < 5%
   - 中规模（n=200-300）：gap < 10%
   - 大规模（n≥400）：gap 可能较高，但仍能找到可行解

2. **多 GPU 收益**：
   - 预期在大规模问题上收益更明显（2-5%）
   - 验证"简化版"在实际场景中的价值

3. **可扩展性**：
   - 观察 gens/s 随规模的变化
   - 识别性能瓶颈（shared memory, L2 cache）

## 实验日期

2026-03-05
