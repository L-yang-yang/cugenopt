/**
 * types.cuh - 核心类型定义
 * 
 * 包含：编码类型、Solution 模板、ProblemConfig/SolverConfig、
 *       SeqRegistry（AOS 序列级权重）、KStepConfig（多步执行）、
 *       RelationMatrix（G/O 关系矩阵）、ProblemBase（CRTP 基类）
 */

#pragma once

// ============================================================
// 编译时常量
// ============================================================
constexpr int MAX_OBJ = 4;    // 最多 4 个目标（16字节，不值得模板化）
constexpr int MAX_SEQ = 32;   // 最大序列数（内置 ~16 + 自定义算子 ≤8，留余量）
constexpr int MAX_K   = 3;    // 多步执行的最大步数（K=1,2,3）
// AOS 权重上下限（归一化后）
constexpr float AOS_WEIGHT_FLOOR = 0.05f;  // 最低权重保底（确保充分探索）
constexpr float AOS_WEIGHT_CAP   = 0.35f;  // 最高权重上限（防止赢者通吃）

// ============================================================
// 枚举类型
// ============================================================

enum class EncodingType {
    Permutation,    // 排列：元素不重复
    Binary,         // 0-1：flip 是主要算子
    Integer         // 有界整数
};

enum class RowMode {
    Single,     // dim1=1，单行（TSP/QAP/Knapsack 等大部分问题）
    Fixed,      // dim1>1，行等长不可变（JSP-Int/Schedule，禁止 SPLIT/MERGE）
    Partition   // dim1>1，元素分区到各行，行长可变（CVRP/VRPTW）
};

enum class ObjDir {
    Minimize,
    Maximize
};

// 多目标比较模式
enum class CompareMode {
    Weighted,       // 加权求和：sum(weight[i] * obj[i])，越小越好
    Lexicographic   // 字典法：按优先级逐目标比较，前面的目标优先
};

enum class MigrateStrategy {
    Ring,       // 环形：各岛最优→邻岛最差（慢传播，高多样性）
    TopN,       // 全局 Top-N 轮转分发（快传播，强收敛）
    Hybrid      // 两者兼顾：Top-N 替换最差 + Ring 替换次差
};


// ============================================================
// SeqID — 统一的 OperationSequence 编号
// ============================================================
// 每个 SeqID 对应一种具体的搜索操作（原子或多步）
// AOS 权重跟踪粒度 = SeqID（每个序列独立权重）
//
// 命名规则：SEQ_{编码}_{操作名}
// 跨编码共享的行级操作统一编号

namespace seq {

// --- Permutation 行内（元素级）---
constexpr int SEQ_PERM_SWAP           = 0;   // swap 两个位置
constexpr int SEQ_PERM_REVERSE        = 1;   // 2-opt（反转区间）
constexpr int SEQ_PERM_INSERT         = 2;   // insert（移动到新位置）
constexpr int SEQ_PERM_3OPT           = 3;   // 3-opt（断 3 边重连）

// --- Permutation 行内（片段级）---
constexpr int SEQ_PERM_OR_OPT         = 4;   // or-opt（移动连续 k 个元素）

// --- Permutation 行内（组合级）---
constexpr int SEQ_PERM_DOUBLE_SWAP    = 30;  // 连续两次 swap（同行）
constexpr int SEQ_PERM_TRIPLE_SWAP    = 31;  // 连续三次 swap（同行）

// --- Permutation 跨行（元素级）---
constexpr int SEQ_PERM_CROSS_RELOCATE = 5;   // 单元素移行
constexpr int SEQ_PERM_CROSS_SWAP     = 6;   // 单元素换行

// --- Permutation 跨行（片段级）---
constexpr int SEQ_PERM_SEG_RELOCATE   = 7;   // 片段移行
constexpr int SEQ_PERM_SEG_SWAP       = 8;   // 片段换行（2-opt*）
constexpr int SEQ_PERM_CROSS_EXCHANGE = 9;   // 片段互换（保序）

// --- Binary 行内（元素级）---
constexpr int SEQ_BIN_FLIP            = 0;   // 翻转一个位
constexpr int SEQ_BIN_SWAP            = 1;   // 交换两个位

// --- Binary 行内（片段级）---
constexpr int SEQ_BIN_SEG_FLIP        = 2;   // 翻转连续 k 个位
constexpr int SEQ_BIN_K_FLIP          = 3;   // 同时翻转 k 个随机位

// --- Binary 跨行 ---
constexpr int SEQ_BIN_CROSS_SWAP      = 4;   // 两行各一个位互换
constexpr int SEQ_BIN_SEG_CROSS_SWAP  = 5;   // 两行各取一段互换

// --- 共享：行级（编码无关）---
constexpr int SEQ_ROW_SWAP            = 10;  // 交换两行
constexpr int SEQ_ROW_REVERSE         = 11;  // 反转行排列
constexpr int SEQ_ROW_SPLIT           = 12;  // 一行拆两行
constexpr int SEQ_ROW_MERGE           = 13;  // 两行合并

// --- 特殊 ---
constexpr int SEQ_PERTURBATION        = 14;  // 扰动（多步不可逆）

// --- Integer 行内（元素级）---
constexpr int SEQ_INT_RANDOM_RESET    = 0;   // 随机一个位置重置为 [lb, ub] 内随机值
constexpr int SEQ_INT_DELTA           = 1;   // 随机一个位置 ±k（clamp 到 [lb, ub]）
constexpr int SEQ_INT_SWAP            = 2;   // 交换两个位置的值

// --- Integer 行内（片段级）---
constexpr int SEQ_INT_SEG_RESET       = 3;   // 连续 k 个位置全部重置
constexpr int SEQ_INT_K_DELTA         = 4;   // 随机 k 个位置各自 ±1

// --- Integer 跨行 ---
constexpr int SEQ_INT_CROSS_SWAP      = 5;   // 两行各一个位置互换

// --- LNS（大邻域搜索）---
constexpr int SEQ_LNS_SEGMENT_SHUFFLE = 20;  // 打乱连续片段
constexpr int SEQ_LNS_SCATTER_SHUFFLE = 21;  // 打乱随机分散位置
constexpr int SEQ_LNS_GUIDED_REBUILD  = 22;  // 关系矩阵引导重建

}  // namespace seq

// ============================================================
// RelationMatrix — G/O 关系矩阵（GPU global memory）
// ============================================================
// G[i][j]: 元素 i 和 j 的分组倾向（对称，越大越倾向同组）
// O[i][j]: 元素 i 排在 j 前面的倾向（不对称）
// 存储为一维数组 [N * N]，行优先
// 小规模 N<200 直接 Dense，P2 再做稀疏化
//
// 更新时机：host 端，每个 batch 间隙
// 使用时机：kernel 中 SEQ_LNS_GUIDED_REBUILD 读取

struct RelationMatrix {
    float* d_G;           // GPU 上的 G 矩阵 [N * N]
    float* d_O;           // GPU 上的 O 矩阵 [N * N]
    float* h_G;           // Host 上的 G 矩阵 [N * N]（用于更新后上传）
    float* h_O;           // Host 上的 O 矩阵 [N * N]
    int    N;             // 元素总数
    float  decay;         // 衰减系数 α（默认 0.95）
    int    update_count;  // 已更新次数（用于冷启动判断）
};

// ============================================================
// SeqRegistry — 运行时可用序列注册表
// ============================================================
// 根据 EncodingType 和 dim1 自动确定哪些序列可用
// 传到 GPU 供 sample_sequence() 使用

enum class SeqCategory : int {
    InRow    = 0,   // 行内算子（swap, reverse, insert, ...）
    CrossRow = 1,   // 跨行算子（cross_relocate, cross_swap, seg_relocate, ...）
    RowLevel = 2,   // 行级算子（row_swap, row_reverse, split, merge）
    LNS      = 3,   // 大邻域搜索
};

struct SeqRegistry {
    int   ids[MAX_SEQ];       // 可用序列的 SeqID 列表
    int   count;              // 可用序列数量
    float weights[MAX_SEQ];   // 每个序列的当前权重（归一化后用于采样）
    float max_w[MAX_SEQ];     // 每个序列的权重上限（0 = 不限，用全局 cap）
    SeqCategory categories[MAX_SEQ];  // 每个序列的分类（约束导向用）
};

// ============================================================
// KStepConfig — 多步执行的步数选择配置
// ============================================================
// K=1: 单步（当前行为），K=2/3: 连续执行多个序列后再评估
// 两层权重体系的第一层
//
// 自适应策略：
//   - 初始 K=1 权重很大（保守），K>1 权重小
//   - K>1 带来改进 → 增大该 K 的权重
//   - 长时间无改进 → 重置/增大 K>1 权重（跳出局部最优）

struct KStepConfig {
    float weights[MAX_K];     // K=1,2,3 的采样权重（归一化）
    int   stagnation_count;   // 连续无改进的 batch 数（用于触发重置）
    int   stagnation_limit;   // 触发重置的阈值（默认 5 个 batch）
};

// 构建默认 K 步配置
inline KStepConfig build_kstep_config() {
    KStepConfig kc;
    kc.weights[0] = 0.80f;   // K=1: 初始主导
    kc.weights[1] = 0.15f;   // K=2: 少量探索
    kc.weights[2] = 0.05f;   // K=3: 极少探索
    kc.stagnation_count = 0;
    kc.stagnation_limit = 5;
    return kc;
};

// ============================================================
// ProblemProfile — 基于结构特征推断的问题画像
// ============================================================
// 第一层：纯结构推断（不感知语义），用于驱动算子注册和初始权重
// 未来第二层：可扩展更细粒度的画像（如多属性、高约束等）

enum class ScaleClass  { Small, Medium, Large };
enum class StructClass { SingleSeq, MultiFixed, MultiPartition };

struct ProblemProfile {
    EncodingType  encoding;
    ScaleClass    scale;
    StructClass   structure;
    float         cross_row_prob;
};

// classify_problem() 定义在 ProblemConfig 之后

// ============================================================
// 权重预设 — 由 ScaleClass 驱动
// ============================================================

struct WeightPreset {
    float w_cubic;
    float w_quadratic;
    float w_lns;
    float lns_cap;
};

inline WeightPreset get_weight_preset(ScaleClass scale) {
    switch (scale) {
        case ScaleClass::Small:  return { 0.50f, 0.80f, 0.006f, 0.01f };
        case ScaleClass::Medium: return { 0.30f, 0.70f, 0.004f, 0.01f };
        case ScaleClass::Large:  return { 0.05f, 0.30f, 0.001f, 0.01f };
    }
    return { 0.50f, 0.80f, 0.006f, 0.01f };
}

// classify_problem() 和 build_seq_registry() 定义在 ProblemConfig 之后

// ============================================================
// Solution<D1, D2> — 解的模板化表示
// ============================================================
// D1: 行数上限 (TSP=1, VRP≤16, Schedule≤8)
// D2: 每行列数上限 (TSP≤64, 背包≤32)
// 每个 Problem 选择最小够用的 D1/D2，编译器生成紧凑的结构

template<int D1, int D2>
struct Solution {
    static constexpr int DIM1 = D1;   // 编译时行数上限
    static constexpr int DIM2 = D2;   // 编译时列数上限
    int   data[D1][D2];               // D1×D2×4 字节
    int   dim2_sizes[D1];             // D1×4 字节
    float objectives[MAX_OBJ];        // 16 字节（固定）
    float penalty;                    // 4 字节
};

// ============================================================
// ProblemConfig — 问题的运行时元信息
// ============================================================

struct ProblemConfig {
    EncodingType encoding;
    int   dim1;                       // 实际使用的行数 (≤ D1)
    int   dim2_default;               // 实际使用的列数 (≤ D2)
    int   num_objectives;
    ObjDir obj_dirs[MAX_OBJ];
    float obj_weights[MAX_OBJ];       // Weighted 模式下的权重
    // 多目标比较
    CompareMode compare_mode = CompareMode::Weighted;
    int   obj_priority[MAX_OBJ] = {0, 1, 2, 3};  // Lexicographic 模式下的比较顺序（索引）
    float obj_tolerance[MAX_OBJ] = {0.0f, 0.0f, 0.0f, 0.0f};  // 字典法容差：差值 <= tol 视为相等
    int   value_lower_bound;
    int   value_upper_bound;
    // v3.4: 统一行模式
    RowMode row_mode      = RowMode::Single;  // 行模式（Single/Fixed/Partition）
    float cross_row_prob  = 0.0f;     // 跨行 move 概率（0=纯行内操作）
    int   total_elements  = 0;        // Partition 模式下的总元素数
    int   perm_repeat_count = 1;      // 排列中每个值的重复次数（1=标准排列，>1=多重集排列）
};

// ============================================================
// SolverConfig — 求解器参数
// ============================================================

struct SolverConfig {
    int   pop_size         = 0;       // 种群大小（0 = 自动匹配 GPU 最大并行度）
    int   max_gen          = 1000;
    float mutation_rate    = 0.1f;
    unsigned seed          = 42;
    bool  verbose          = true;
    int   print_every      = 100;
    // 岛屿模型参数
    int   num_islands      = 1;       // 0 = 自适应，1 = 纯爬山（无岛屿），>1 = 岛屿模型
    int   migrate_interval = 100;     // 每隔多少代执行一次迁移
    MigrateStrategy migrate_strategy = MigrateStrategy::Hybrid;
    // 模拟退火参数
    float sa_temp_init     = 0.0f;    // 初始温度（0 = 禁用 SA，纯爬山）
    float sa_alpha         = 0.998f;  // 冷却率（每代乘以 alpha）
    // v1.0: 交叉参数
    float crossover_rate   = 0.1f;    // 每代中执行交叉的概率（vs 变异）
    // v2.0: 自适应算子选择
    bool  use_aos          = false;   // 启用 AOS（batch 间更新算子权重）
    float aos_weight_floor = AOS_WEIGHT_FLOOR;  // 运行时可覆盖的 floor
    float aos_weight_cap   = AOS_WEIGHT_CAP;    // 运行时可覆盖的 cap
    // v2.1: 初始解策略
    int   init_oversample  = 4;       // 采样倍数（1 = 不做采样择优，即纯随机）
    float init_random_ratio = 0.3f;   // 纯随机解占比（多样性保底）
    // v3.0: 工程可用性
    float time_limit_sec   = 0.0f;   // 时间限制（秒，0 = 不限制，按 max_gen 跑完）
    int   stagnation_limit = 0;      // 收敛检测：连续多少个 batch 无改进后 reheat（0 = 禁用）
    float reheat_ratio     = 0.5f;   // reheat 时温度恢复到初始温度的比例
    // v3.5: CUDA Graph
    bool  use_cuda_graph   = false;  // 启用 CUDA Graph（减少 kernel launch 开销）
    // v3.6: AOS 更新频率控制
    int   aos_update_interval = 10;  // 每隔多少个 batch 更新一次 AOS 权重（降低 cudaMemcpy 同步频率）
    // v4.0: 约束导向 + 分层搜索
    bool  use_constraint_directed = false;  // 启用约束导向（根据 penalty 比例动态调整跨行算子权重）
    bool  use_phased_search       = false;  // 启用分层搜索（按进度调整全局 floor/cap）
    // 分层搜索参数：三期阈值
    float phase_explore_end  = 0.30f;  // 探索期结束（进度比例）
    float phase_refine_start = 0.70f;  // 精细期开始（进度比例）
    // 约束导向参数
    float constraint_boost_max = 2.5f; // 高约束时跨行算子 cap 提升倍率上限
};

// ============================================================
// classify_problem — 从 ProblemConfig 推断问题画像
// ============================================================

inline ProblemProfile classify_problem(const ProblemConfig& pcfg) {
    ProblemProfile p;
    p.encoding = pcfg.encoding;

    if      (pcfg.dim2_default <= 100) p.scale = ScaleClass::Small;
    else if (pcfg.dim2_default <= 250) p.scale = ScaleClass::Medium;
    else                               p.scale = ScaleClass::Large;

    if (pcfg.dim1 <= 1)
        p.structure = StructClass::SingleSeq;
    else if (pcfg.row_mode == RowMode::Partition)
        p.structure = StructClass::MultiPartition;
    else
        p.structure = StructClass::MultiFixed;

    p.cross_row_prob = pcfg.cross_row_prob;
    return p;
}

// ============================================================
// build_seq_registry — 由 ProblemProfile 驱动的算子注册
// ============================================================

inline SeqRegistry build_seq_registry(const ProblemProfile& prof) {
    SeqRegistry reg;
    reg.count = 0;
    for (int i = 0; i < MAX_SEQ; i++) {
        reg.ids[i] = -1; reg.weights[i] = 0.0f;
        reg.max_w[i] = 0.0f; reg.categories[i] = SeqCategory::InRow;
    }

    auto add = [&](int id, float w, SeqCategory cat, float cap = 0.0f) {
        if (reg.count >= MAX_SEQ) return;
        reg.ids[reg.count] = id;
        reg.weights[reg.count] = w;
        reg.max_w[reg.count] = cap;
        reg.categories[reg.count] = cat;
        reg.count++;
    };

    WeightPreset wp = get_weight_preset(prof.scale);
    bool multi_row = (prof.structure != StructClass::SingleSeq);
    float cr = prof.cross_row_prob;

    if (prof.encoding == EncodingType::Permutation) {
        add(seq::SEQ_PERM_SWAP,    1.0f, SeqCategory::InRow);
        add(seq::SEQ_PERM_REVERSE, 1.0f, SeqCategory::InRow);
        add(seq::SEQ_PERM_INSERT,  1.0f, SeqCategory::InRow);
        add(seq::SEQ_PERM_DOUBLE_SWAP, 0.5f, SeqCategory::InRow);
        add(seq::SEQ_PERM_TRIPLE_SWAP, 0.3f, SeqCategory::InRow);

        add(seq::SEQ_PERM_3OPT,   wp.w_cubic,     SeqCategory::InRow);
        add(seq::SEQ_PERM_OR_OPT, wp.w_quadratic,  SeqCategory::InRow);

        if (multi_row && cr > 0.0f) {
            add(seq::SEQ_PERM_CROSS_RELOCATE, 0.6f * cr, SeqCategory::CrossRow);
            add(seq::SEQ_PERM_CROSS_SWAP,     0.6f * cr, SeqCategory::CrossRow);
            add(seq::SEQ_PERM_SEG_RELOCATE,   0.5f * cr, SeqCategory::CrossRow);
            add(seq::SEQ_PERM_SEG_SWAP,       0.5f * cr, SeqCategory::CrossRow);
            add(seq::SEQ_PERM_CROSS_EXCHANGE,  0.4f * cr, SeqCategory::CrossRow);
        }
        if (multi_row) {
            add(seq::SEQ_ROW_SWAP,    0.3f, SeqCategory::RowLevel);
            add(seq::SEQ_ROW_REVERSE, 0.2f, SeqCategory::RowLevel);
            if (prof.structure == StructClass::MultiPartition) {
                add(seq::SEQ_ROW_SPLIT,  0.2f, SeqCategory::RowLevel);
                add(seq::SEQ_ROW_MERGE,  0.2f, SeqCategory::RowLevel);
            }
        }
        add(seq::SEQ_LNS_SEGMENT_SHUFFLE, wp.w_lns, SeqCategory::LNS, wp.lns_cap);
        add(seq::SEQ_LNS_SCATTER_SHUFFLE, wp.w_lns, SeqCategory::LNS, wp.lns_cap);
        add(seq::SEQ_LNS_GUIDED_REBUILD,  wp.w_lns, SeqCategory::LNS, wp.lns_cap);
    }
    else if (prof.encoding == EncodingType::Binary) {
        add(seq::SEQ_BIN_FLIP, 1.0f, SeqCategory::InRow);
        add(seq::SEQ_BIN_SWAP, 0.8f, SeqCategory::InRow);
        add(seq::SEQ_BIN_SEG_FLIP, 0.6f, SeqCategory::InRow);
        add(seq::SEQ_BIN_K_FLIP,   0.6f, SeqCategory::InRow);
        if (multi_row && cr > 0.0f) {
            add(seq::SEQ_BIN_CROSS_SWAP,     0.5f * cr, SeqCategory::CrossRow);
            add(seq::SEQ_BIN_SEG_CROSS_SWAP, 0.4f * cr, SeqCategory::CrossRow);
        }
        if (multi_row) {
            add(seq::SEQ_ROW_SWAP,    0.3f, SeqCategory::RowLevel);
            add(seq::SEQ_ROW_REVERSE, 0.2f, SeqCategory::RowLevel);
            if (prof.structure == StructClass::MultiPartition) {
                add(seq::SEQ_ROW_SPLIT,  0.2f, SeqCategory::RowLevel);
                add(seq::SEQ_ROW_MERGE,  0.2f, SeqCategory::RowLevel);
            }
        }
    }
    else if (prof.encoding == EncodingType::Integer) {
        add(seq::SEQ_INT_RANDOM_RESET, 1.0f, SeqCategory::InRow);
        add(seq::SEQ_INT_DELTA,        1.0f, SeqCategory::InRow);
        add(seq::SEQ_INT_SWAP,         0.8f, SeqCategory::InRow);
        add(seq::SEQ_INT_SEG_RESET,    0.6f, SeqCategory::InRow);
        add(seq::SEQ_INT_K_DELTA,      0.6f, SeqCategory::InRow);
        if (multi_row && cr > 0.0f) {
            add(seq::SEQ_INT_CROSS_SWAP, 0.5f * cr, SeqCategory::CrossRow);
        }
        if (multi_row) {
            add(seq::SEQ_ROW_SWAP,    0.3f, SeqCategory::RowLevel);
            add(seq::SEQ_ROW_REVERSE, 0.2f, SeqCategory::RowLevel);
            if (prof.structure == StructClass::MultiPartition) {
                add(seq::SEQ_ROW_SPLIT,  0.2f, SeqCategory::RowLevel);
                add(seq::SEQ_ROW_MERGE,  0.2f, SeqCategory::RowLevel);
            }
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < reg.count; i++) sum += reg.weights[i];
    if (sum > 0.0f) {
        for (int i = 0; i < reg.count; i++) reg.weights[i] /= sum;
    }
    return reg;
}

// ============================================================
// ObjConfig — 传到 GPU 的目标比较配置（紧凑结构）
// ============================================================

struct ObjConfig {
    int         num_obj;
    CompareMode mode;
    ObjDir      dirs[MAX_OBJ];       // 每个目标的方向
    float       weights[MAX_OBJ];    // Weighted 模式下的权重
    int         priority[MAX_OBJ];   // Lexicographic 模式下的比较顺序
    float       tolerance[MAX_OBJ];  // Lexicographic 模式下的容差
};

// 从 ProblemConfig 构造 ObjConfig（CPU 端）
inline ObjConfig make_obj_config(const ProblemConfig& pcfg) {
    ObjConfig oc;
    oc.num_obj = pcfg.num_objectives;
    oc.mode = pcfg.compare_mode;
    for (int i = 0; i < MAX_OBJ; i++) {
        oc.dirs[i]      = pcfg.obj_dirs[i];
        oc.weights[i]   = pcfg.obj_weights[i];
        oc.priority[i]  = pcfg.obj_priority[i];
        oc.tolerance[i] = pcfg.obj_tolerance[i];
    }
    return oc;
}

// ============================================================
// SolveResult — solve() 的返回值
// ============================================================

enum class StopReason { MaxGen, TimeLimit, Stagnation };

template<typename Sol>
struct SolveResult {
    Sol         best_solution;
    float       elapsed_ms     = 0.0f;
    int         generations    = 0;
    StopReason  stop_reason    = StopReason::MaxGen;
};

// ============================================================
// 目标重要性映射 — 统一 Weighted / Lexicographic 的重要性度量
// ============================================================
// 用于初始化选种（NSGA-II 加权拥挤度 + 核心目标预留名额）
// Weighted:      importance[i] = weight[i] / Σweight
// Lexicographic: importance[i] = 0.5^rank[i] / Σ(0.5^rank)
//   → 第一优先级 ~57%，第二 ~29%，第三 ~14%

inline void compute_importance(const ObjConfig& oc, float* importance) {
    float sum = 0.0f;
    for (int i = 0; i < oc.num_obj; i++) {
        if (oc.mode == CompareMode::Weighted) {
            importance[i] = oc.weights[i];
        } else {
            int rank = oc.priority[i];
            importance[i] = 1.0f;
            for (int r = 0; r < rank; r++) importance[i] *= 0.5f;  // 0.5^rank
        }
        sum += importance[i];
    }
    if (sum > 0.0f) {
        for (int i = 0; i < oc.num_obj; i++)
            importance[i] /= sum;
    }
}

// ============================================================
// 比较工具 — 支持 Weighted / Lexicographic
// ============================================================

// 将目标值统一为"越小越好"：Maximize 目标取负
__device__ __host__ inline float normalize_obj(float val, ObjDir dir) {
    return (dir == ObjDir::Maximize) ? -val : val;
}

// 核心比较：a 是否优于 b
template<typename Sol>
__device__ inline bool is_better(const Sol& a, const Sol& b,
                                  const ObjConfig& oc) {
    // penalty 优先：可行解一定优于不可行解
    if (a.penalty <= 0.0f && b.penalty > 0.0f) return true;
    if (a.penalty > 0.0f && b.penalty <= 0.0f) return false;
    if (a.penalty > 0.0f && b.penalty > 0.0f) return a.penalty < b.penalty;
    
    if (oc.mode == CompareMode::Weighted) {
        // 加权求和（权重已包含方向信息：Maximize 目标用负权重，或由 normalize_obj 处理）
        float sum_a = 0.0f, sum_b = 0.0f;
        for (int i = 0; i < oc.num_obj; i++) {
            float na = normalize_obj(a.objectives[i], oc.dirs[i]);
            float nb = normalize_obj(b.objectives[i], oc.dirs[i]);
            sum_a += oc.weights[i] * na;
            sum_b += oc.weights[i] * nb;
        }
        return sum_a < sum_b;
    } else {
        // 字典法：按 priority 顺序逐目标比较
        for (int p = 0; p < oc.num_obj; p++) {
            int idx = oc.priority[p];
            float va = normalize_obj(a.objectives[idx], oc.dirs[idx]);
            float vb = normalize_obj(b.objectives[idx], oc.dirs[idx]);
            float diff = va - vb;
            if (diff < -oc.tolerance[idx]) return true;   // a 明显更好
            if (diff >  oc.tolerance[idx]) return false;  // b 明显更好
            // 在容差内视为相等 → 继续比较下一个目标
        }
        return false;  // 所有目标都在容差内相等
    }
}

// 标量化（SA 接受概率用）：返回越小越好的标量
template<typename Sol>
__device__ __host__ inline float scalar_objective(const Sol& sol,
                                                    const ObjConfig& oc) {
    if (oc.mode == CompareMode::Weighted) {
        float sum = 0.0f;
        for (int i = 0; i < oc.num_obj; i++)
            sum += oc.weights[i] * normalize_obj(sol.objectives[i], oc.dirs[i]);
        return sum;
    } else {
        // 字典法下 SA 用第一优先级目标作为标量
        int idx = oc.priority[0];
        return normalize_obj(sol.objectives[idx], oc.dirs[idx]);
    }
}

// 轻量比较：直接操作 float[] 目标数组（避免复制整个 Sol）
__device__ inline bool obj_is_better(const float* new_objs, const float* old_objs,
                                      const ObjConfig& oc) {
    if (oc.mode == CompareMode::Weighted) {
        float sum_new = 0.0f, sum_old = 0.0f;
        for (int i = 0; i < oc.num_obj; i++) {
            sum_new += oc.weights[i] * normalize_obj(new_objs[i], oc.dirs[i]);
            sum_old += oc.weights[i] * normalize_obj(old_objs[i], oc.dirs[i]);
        }
        return sum_new < sum_old;
    } else {
        for (int p = 0; p < oc.num_obj; p++) {
            int idx = oc.priority[p];
            float va = normalize_obj(new_objs[idx], oc.dirs[idx]);
            float vb = normalize_obj(old_objs[idx], oc.dirs[idx]);
            float diff = va - vb;
            if (diff < -oc.tolerance[idx]) return true;
            if (diff >  oc.tolerance[idx]) return false;
        }
        return false;
    }
}

// 轻量标量化：直接操作 float[] 目标数组
__device__ __host__ inline float obj_scalar(const float* objs, const ObjConfig& oc) {
    if (oc.mode == CompareMode::Weighted) {
        float sum = 0.0f;
        for (int i = 0; i < oc.num_obj; i++)
            sum += oc.weights[i] * normalize_obj(objs[i], oc.dirs[i]);
        return sum;
    } else {
        int idx = oc.priority[0];
        return normalize_obj(objs[idx], oc.dirs[idx]);
    }
}

// ============================================================
// AOSStats — 自适应算子选择统计（每个 block 一份）
// ============================================================
// v3.0: 粒度从 3 层 → MAX_SEQ 个序列
// 记录每个序列的使用次数和改进次数
// batch 结束后由 host 聚合，更新 SeqRegistry 权重

struct AOSStats {
    // 算子层统计（第二层）
    int usage[MAX_SEQ];       // 各序列使用次数
    int improvement[MAX_SEQ]; // 各序列改进次数（delta < 0 且被接受）
    // K 步数层统计（第一层）
    int k_usage[MAX_K];       // K=1,2,3 各自使用次数
    int k_improvement[MAX_K]; // K=1,2,3 各自改进次数
};

// ============================================================
// ObjDef — 单个目标的定义（编译期常量）
// ============================================================

struct ObjDef {
    ObjDir dir;           // 优化方向
    float  weight;        // Weighted 模式下的权重
    float  tolerance;     // Lexicographic 模式下的容差
};

// ============================================================
// HeuristicMatrix — 启发式初始解构造用的数据矩阵描述
// ============================================================

struct HeuristicMatrix {
    const float* data;   // host 端 N*N 矩阵
    int N;               // 维度
};

// ============================================================
// ProblemBase<Derived, D1, D2> — CRTP 基类
//
// 用户继承此基类，提供：
//   static constexpr ObjDef OBJ_DEFS[] = {...};   — 目标元信息
//   __device__ float compute_obj(int idx, ...) const;  — 目标分发
//   __device__ float compute_penalty(...) const;
//
// 约定：OBJ_DEFS 和 compute_obj 紧挨着写，case N 对应 OBJ_DEFS[N]
// NUM_OBJ 由 sizeof(OBJ_DEFS) 自动推导，无需手动维护
//
// 基类自动提供：
//   evaluate(sol)           — 遍历目标列表调用 compute_obj
//   fill_obj_config(cfg)    — 从 OBJ_DEFS 自动填充 ProblemConfig
//   obj_config()            — 直接生成 ObjConfig
// ============================================================

template<typename Derived, int D1_, int D2_>
struct ProblemBase {
    static constexpr int D1 = D1_;
    static constexpr int D2 = D2_;
    using Sol = Solution<D1, D2>;
    
    // NUM_OBJ 从 OBJ_DEFS 数组自动推导
    static constexpr int NUM_OBJ = sizeof(Derived::OBJ_DEFS) / sizeof(ObjDef);
    
    // 自动评估：遍历目标列表
    __device__ void evaluate(Sol& sol) const {
        const auto& self = static_cast<const Derived&>(*this);
        constexpr int n = sizeof(Derived::OBJ_DEFS) / sizeof(ObjDef);
        for (int i = 0; i < n; i++)
            sol.objectives[i] = self.compute_obj(i, sol);
        sol.penalty = self.compute_penalty(sol);
    }
    
    // 从 OBJ_DEFS 自动填充 ProblemConfig 的目标部分
    void fill_obj_config(ProblemConfig& cfg) const {
        constexpr int n = sizeof(Derived::OBJ_DEFS) / sizeof(ObjDef);
        cfg.num_objectives = n;
        for (int i = 0; i < n; i++) {
            cfg.obj_dirs[i]      = Derived::OBJ_DEFS[i].dir;
            cfg.obj_weights[i]   = Derived::OBJ_DEFS[i].weight;
            cfg.obj_tolerance[i] = Derived::OBJ_DEFS[i].tolerance;
            cfg.obj_priority[i]  = i;  // 列表顺序即优先级
        }
    }
    
    // 直接生成 ObjConfig（供 solver 使用）
    ObjConfig obj_config() const {
        ProblemConfig pcfg;
        fill_obj_config(pcfg);
        return make_obj_config(pcfg);
    }
    
    // 每个 block 在 global memory 中的热数据工作集大小（字节）
    // 用于 auto pop_size 估算 L2 cache 压力
    // 默认 = shared_mem_bytes()（数据在 smem 时，gmem 工作集为 0 不影响）
    // 子类覆盖：当 shared_mem_bytes() 返回 0（数据放不进 smem）时，
    //           返回实际数据大小（如距离矩阵 n*n*sizeof(float)）
    size_t working_set_bytes() const {
        return static_cast<const Derived&>(*this).shared_mem_bytes();
    }
    
    // 可选：初始化 G/O 关系矩阵（为 GUIDED_REBUILD 提供先验知识）
    // G[i*N+j]: 元素 i 和 j 的分组倾向（对称，[0,1]，越大越倾向同组）
    // O[i*N+j]: 元素 i 排在 j 前面的倾向（不对称，[0,1]）
    // 默认不提供（全零），搜索过程中通过 EMA 从历史好解积累
    // 用户覆盖示例：距离近 → G 和 O 都高
    void init_relation_matrix(float* h_G, float* h_O, int N) const {
        (void)h_G; (void)h_O; (void)N;  // 默认：不做任何事（保持全零）
    }
    
    // 可选：返回 host 端数据矩阵供启发式初始解构造
    // 默认返回 0（不提供），子类 override 后填充 out 数组并返回实际数量
    int heuristic_matrices(HeuristicMatrix* out, int max_count) const {
        (void)out; (void)max_count;
        return 0;
    }
};
