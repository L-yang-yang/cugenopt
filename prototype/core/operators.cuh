/**
 * operators.cuh - 四层搜索算子体系（Device 端）
 *
 * v1.0: 二维通用编码的完整算子层次
 *
 * 层次结构（所有算子只看 data[D1][D2] + dim2_sizes，不感知问题语义）：
 *
 *   第 1 层 - 元素级（Element）: 操作单个元素
 *     行内: swap, reverse(2-opt), insert, flip
 *     跨行: cross_relocate（单元素移行）, cross_swap（单元素换行）
 *
 *   第 2 层 - 片段级（Segment）: 操作连续片段
 *     行内: or_opt（移动连续 k 个元素到行内新位置）
 *     跨行: seg_relocate（片段从一行移到另一行）
 *            seg_swap（两行各取一段互换，即 2-opt*）
 *
 *   第 3 层 - 行级（Row）: 操作整行
 *     row_swap（交换两行全部内容和长度）
 *     row_reverse（反转行的排列顺序）
 *     row_split（一行拆成两行）
 *     row_merge（两行合并为一行）
 *
 *   第 4 层 - 交叉（Crossover）: 组合两个解
 *     row_crossover（从父代 A/B 各取若干行组成子代）
 *     uniform_crossover（逐元素从两个父代中选）
 *
 * Move 描述符：
 *   row, row2: 行索引（row2=-1 表示行内）
 *   op:        操作码
 *   pos1, pos2: 位置参数
 *   seg_len:   片段长度（第 2 层使用）
 *
 * 设计原则：
 *   - 所有算子对问题类型无感知，只操作二维数组
 *   - 每个算子都有对应的 undo 操作
 *   - 空行安全：自动降级为 no-op
 *   - 编码类型决定可用算子集
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

namespace ops {

// ============================================================
// Op 码常量 — 按层次编号，避免冲突
// ============================================================

// 通用
constexpr int OP_NOOP             = -1;

// --- 第 1 层：元素级 ---
// Permutation 行内
constexpr int PERM_SWAP           = 0;   // 交换两个位置
constexpr int PERM_REVERSE        = 1;   // 反转区间（2-opt）
constexpr int PERM_INSERT         = 2;   // 移动单个元素到新位置
// Permutation 跨行
constexpr int PERM_CROSS_RELOCATE = 3;   // 单元素从一行移到另一行
constexpr int PERM_CROSS_SWAP     = 4;   // 两行各一个元素互换
// Binary 行内
constexpr int BIN_FLIP            = 0;   // 翻转一个位
constexpr int BIN_SWAP            = 1;   // 交换两个位
// Binary 跨行
constexpr int BIN_CROSS_SWAP      = 2;   // 两行各一个位互换

// --- 第 1 层（续）：排列行内 ---
constexpr int PERM_3OPT           = 5;   // 3-opt：断 3 条边重连

// --- 第 2 层：片段级 ---
constexpr int PERM_OR_OPT         = 10;  // 行内：移动连续 k 个元素
constexpr int PERM_SEG_RELOCATE   = 11;  // 跨行：片段从一行移到另一行
constexpr int PERM_SEG_SWAP       = 12;  // 跨行：两行各取一段互换（2-opt*）
constexpr int PERM_CROSS_EXCHANGE = 15;  // 跨行：两行各取一段互换（保持各自内部顺序）
constexpr int BIN_SEG_FLIP        = 13;  // 行内：翻转连续 k 个位
constexpr int BIN_SEG_CROSS_SWAP  = 14;  // 跨行：两行各取一段互换
constexpr int BIN_K_FLIP          = 16;  // 行内：同时翻转 k 个随机位

// --- 第 3 层：行级 ---
constexpr int ROW_SWAP            = 20;  // 交换两行全部内容
constexpr int ROW_REVERSE         = 21;  // 反转行的排列顺序（行号重排）
constexpr int ROW_SPLIT           = 22;  // 一行拆成两行
constexpr int ROW_MERGE           = 23;  // 两行合并为一行

// --- 特殊：扰动（连续多步 move，不可 undo，用于跳出局部最优）---
constexpr int PERTURBATION        = 40;

// --- 第 4 层：交叉 ---
constexpr int CROSS_ROW           = 30;  // 行级交叉：从两个父代各取若干行
constexpr int CROSS_UNIFORM       = 31;  // 均匀交叉：逐元素从两个父代选

// ============================================================
// Move 描述符 — 编码级别的变动描述
// ============================================================

struct Move {
    int row;            // 源行（或第一行）
    int row2;           // 目标行（-1 = 行内）
    int op;             // 操作码
    int pos1, pos2;     // 位置参数
    int seg_len;        // 片段长度（第 2 层使用，其他层 = 0）
};

}  // namespace ops

namespace ops {

// ============================================================
// 第 1 层：元素级底层操作
// ============================================================

// --- Permutation 行内 ---

__device__ inline void perm_swap(int* row, int i, int j) {
    int tmp = row[i]; row[i] = row[j]; row[j] = tmp;
}

__device__ inline void perm_reverse(int* row, int i, int j) {
    while (i < j) { perm_swap(row, i, j); i++; j--; }
}

__device__ inline void perm_insert(int* row, int from, int to, int size) {
    int val = row[from];
    if (from < to) { for (int k = from; k < to; k++) row[k] = row[k+1]; }
    else           { for (int k = from; k > to; k--) row[k] = row[k-1]; }
    row[to] = val;
}

// --- Permutation 跨行 ---

/// cross_relocate: 从 src_row[src_pos] 取出元素，插入 dst_row[dst_pos]
__device__ inline void perm_cross_relocate(int* src_row, int& src_size,
                                            int* dst_row, int& dst_size,
                                            int src_pos, int dst_pos) {
    int val = src_row[src_pos];
    for (int k = src_pos; k < src_size - 1; k++)
        src_row[k] = src_row[k + 1];
    src_size--;
    for (int k = dst_size; k > dst_pos; k--)
        dst_row[k] = dst_row[k - 1];
    dst_row[dst_pos] = val;
    dst_size++;
}

/// cross_swap: 交换 rowA[posA] 和 rowB[posB]
__device__ inline void cross_swap_elem(int* rowA, int posA, int* rowB, int posB) {
    int tmp = rowA[posA]; rowA[posA] = rowB[posB]; rowB[posB] = tmp;
}

// --- Permutation 行内：3-opt ---
// 断开 3 条边，选择最佳重连方式（共 8 种组合，取随机一种非恒等变换）
// 参数：3 个断点 i < j < k，将路线分为 seg0=[0,i] seg1=[i+1,j] seg2=[j+1,k] seg3=[k+1,end]
// 实现：随机选一种重连（reverse seg1, reverse seg2, 或两者都反转）
// pos1=i, pos2=j, seg_len 编码 k
__device__ inline void perm_3opt(int* row, int size, int i, int j, int k) {
    // 3-opt 有多种重连方式，这里实现最常用的 3 种非恒等变换：
    //   type 1: reverse [i+1, j]                    — 等价于 2-opt(i+1, j)
    //   type 2: reverse [j+1, k]                    — 等价于 2-opt(j+1, k)
    //   type 3: reverse [i+1, j] + reverse [j+1, k] — 真正的 3-opt move
    //   type 4: 将 seg1 和 seg2 互换位置（不反转）  — or-opt 的泛化
    // 我们随机选 type 3 或 type 4（type 1/2 已被 2-opt 覆盖）
    // 这里固定做 type 3（双反转），因为它是 2-opt 无法达到的唯一新邻域
    // reverse [i+1, j]
    int lo = i + 1, hi = j;
    while (lo < hi) { int t = row[lo]; row[lo] = row[hi]; row[hi] = t; lo++; hi--; }
    // reverse [j+1, k]
    lo = j + 1; hi = k;
    while (lo < hi) { int t = row[lo]; row[lo] = row[hi]; row[hi] = t; lo++; hi--; }
}

// 3-opt undo: 再做一次相同操作即可恢复（双反转是自反的）
__device__ inline void perm_3opt_undo(int* row, int size, int i, int j, int k) {
    perm_3opt(row, size, i, j, k);  // 自反
}

// --- Binary 行内 ---

__device__ inline void bin_flip(int* row, int i) { row[i] = 1 - row[i]; }

__device__ inline void bin_swap(int* row, int i, int j) {
    int tmp = row[i]; row[i] = row[j]; row[j] = tmp;
}

// ============================================================
// 第 2 层：片段级底层操作
// ============================================================

/// or_opt: 行内移动连续 seg_len 个元素（从 from 开始）到 to 位置
/// 等价于：取出 [from, from+seg_len)，插入到 to 之前
/// 约束：from + seg_len <= size, to 不在 [from, from+seg_len) 内
__device__ inline void perm_or_opt(int* row, int size, int from, int to, int seg_len) {
    // 临时缓冲（片段最大长度受限于寄存器，实际 seg_len 通常 <= 4）
    int buf[8];  // 足够覆盖常见 seg_len
    int actual_len = (seg_len > 8) ? 8 : seg_len;
    
    // 保存片段
    for (int i = 0; i < actual_len; i++) buf[i] = row[from + i];
    
    // 移除片段（左移填补空洞）
    int new_size = size - actual_len;
    for (int k = from; k < new_size; k++) row[k] = row[k + actual_len];
    
    // 计算插入位置（移除后的坐标系）
    int ins = (to > from) ? to - actual_len : to;
    if (ins < 0) ins = 0;
    if (ins > new_size) ins = new_size;
    
    // 插入片段（右移腾位）
    for (int k = new_size - 1; k >= ins; k--) row[k + actual_len] = row[k];
    for (int i = 0; i < actual_len; i++) row[ins + i] = buf[i];
}

/// seg_relocate: 从 src_row 取出连续 seg_len 个元素，插入 dst_row 的 dst_pos
/// src_size 减 seg_len，dst_size 加 seg_len
__device__ inline void perm_seg_relocate(int* src_row, int& src_size,
                                          int* dst_row, int& dst_size,
                                          int src_pos, int dst_pos, int seg_len) {
    int buf[8];
    int actual_len = (seg_len > 8) ? 8 : seg_len;
    
    // 保存片段
    for (int i = 0; i < actual_len; i++) buf[i] = src_row[src_pos + i];
    
    // 源行：移除（左移）
    for (int k = src_pos; k < src_size - actual_len; k++)
        src_row[k] = src_row[k + actual_len];
    src_size -= actual_len;
    
    // 目标行：插入（右移）
    for (int k = dst_size - 1; k >= dst_pos; k--)
        dst_row[k + actual_len] = dst_row[k];
    for (int i = 0; i < actual_len; i++)
        dst_row[dst_pos + i] = buf[i];
    dst_size += actual_len;
}

/// seg_swap: 两行各取一段互换（2-opt* 的通用形式）
/// rowA[posA..posA+lenA) <-> rowB[posB..posB+lenB)
/// 行长变化：sizeA += (lenB - lenA), sizeB += (lenA - lenB)
__device__ inline void perm_seg_swap(int* rowA, int& sizeA, int posA, int lenA,
                                      int* rowB, int& sizeB, int posB, int lenB) {
    int bufA[8], bufB[8];
    int aLen = (lenA > 8) ? 8 : lenA;
    int bLen = (lenB > 8) ? 8 : lenB;
    
    // 保存两段
    for (int i = 0; i < aLen; i++) bufA[i] = rowA[posA + i];
    for (int i = 0; i < bLen; i++) bufB[i] = rowB[posB + i];
    
    // 从 rowA 移除 segA，腾出空间插入 segB
    // 先移除
    int newSizeA = sizeA - aLen;
    for (int k = posA; k < newSizeA; k++) rowA[k] = rowA[k + aLen];
    // 再插入 segB
    for (int k = newSizeA - 1; k >= posA; k--) rowA[k + bLen] = rowA[k];
    for (int i = 0; i < bLen; i++) rowA[posA + i] = bufB[i];
    sizeA = newSizeA + bLen;
    
    // 从 rowB 移除 segB，腾出空间插入 segA
    int newSizeB = sizeB - bLen;
    for (int k = posB; k < newSizeB; k++) rowB[k] = rowB[k + bLen];
    for (int k = newSizeB - 1; k >= posB; k--) rowB[k + aLen] = rowB[k];
    for (int i = 0; i < aLen; i++) rowB[posB + i] = bufA[i];
    sizeB = newSizeB + aLen;
}

/// cross_exchange: 两行各取一段互换，保持各自内部顺序
/// 与 seg_swap 的区别：seg_swap 是等长互换，cross_exchange 允许不等长
/// rowA[posA..posA+lenA) <-> rowB[posB..posB+lenB)
/// 行长变化：sizeA += (lenB - lenA), sizeB += (lenA - lenB)
__device__ inline void perm_cross_exchange(int* rowA, int& sizeA, int posA, int lenA,
                                            int* rowB, int& sizeB, int posB, int lenB) {
    int bufA[8], bufB[8];
    int aLen = (lenA > 8) ? 8 : lenA;
    int bLen = (lenB > 8) ? 8 : lenB;
    
    for (int i = 0; i < aLen; i++) bufA[i] = rowA[posA + i];
    for (int i = 0; i < bLen; i++) bufB[i] = rowB[posB + i];
    
    // rowA: 移除 segA，插入 segB
    int newSizeA = sizeA - aLen;
    for (int k = posA; k < newSizeA; k++) rowA[k] = rowA[k + aLen];
    for (int k = newSizeA - 1; k >= posA; k--) rowA[k + bLen] = rowA[k];
    for (int i = 0; i < bLen; i++) rowA[posA + i] = bufB[i];
    sizeA = newSizeA + bLen;
    
    // rowB: 移除 segB，插入 segA
    int newSizeB = sizeB - bLen;
    for (int k = posB; k < newSizeB; k++) rowB[k] = rowB[k + bLen];
    for (int k = newSizeB - 1; k >= posB; k--) rowB[k + aLen] = rowB[k];
    for (int i = 0; i < aLen; i++) rowB[posB + i] = bufA[i];
    sizeB = newSizeB + aLen;
}

/// k-bit flip: 同时翻转 k 个随机位（Binary 编码）
/// positions 数组存储要翻转的位置，k = 实际翻转数
__device__ inline void bin_k_flip(int* row, int size, int k, curandState* rng) {
    for (int i = 0; i < k; i++) {
        int pos = rand_int(rng, size);
        row[pos] = 1 - row[pos];
    }
}

/// seg_flip: 翻转行内连续 seg_len 个位（Binary 编码）
__device__ inline void bin_seg_flip(int* row, int pos, int seg_len) {
    for (int i = 0; i < seg_len; i++) row[pos + i] = 1 - row[pos + i];
}

/// seg_cross_swap: 两行各取一段互换（Binary 编码，等长）
__device__ inline void bin_seg_cross_swap(int* rowA, int posA,
                                           int* rowB, int posB, int seg_len) {
    for (int i = 0; i < seg_len; i++) {
        int tmp = rowA[posA + i];
        rowA[posA + i] = rowB[posB + i];
        rowB[posB + i] = tmp;
    }
}

// ============================================================
// Integer 编码底层操作
// ============================================================

/// int_clamp: 将值限制在 [lb, ub] 范围内
__device__ inline int int_clamp(int v, int lb, int ub) {
    if (v < lb) return lb;
    if (v > ub) return ub;
    return v;
}

/// int_random_reset: 随机一个位置重置为 [lb, ub] 内随机值
__device__ inline void int_random_reset(int* row, int pos, int lb, int ub,
                                         curandState* rng) {
    row[pos] = lb + (curand(rng) % (ub - lb + 1));
}

/// int_delta: 随机一个位置 ±k（clamp 到 [lb, ub]）
__device__ inline void int_delta(int* row, int pos, int lb, int ub,
                                  curandState* rng) {
    int range = ub - lb + 1;
    int max_step = (range < 5) ? range : 5;
    int step = 1 + (curand(rng) % max_step);
    if (curand(rng) & 1) step = -step;
    row[pos] = int_clamp(row[pos] + step, lb, ub);
}

/// int_seg_reset: 连续 k 个位置全部重置为 [lb, ub] 内随机值
__device__ inline void int_seg_reset(int* row, int pos, int seg_len,
                                      int lb, int ub, curandState* rng) {
    int range = ub - lb + 1;
    for (int i = 0; i < seg_len; i++)
        row[pos + i] = lb + (curand(rng) % range);
}

/// int_k_delta: 随机 k 个位置各自 ±1
__device__ inline void int_k_delta(int* row, int size, int k,
                                    int lb, int ub, curandState* rng) {
    for (int i = 0; i < k; i++) {
        int pos = rand_int(rng, size);
        int step = (curand(rng) & 1) ? 1 : -1;
        row[pos] = int_clamp(row[pos] + step, lb, ub);
    }
}

// ============================================================
// 第 3 层：行级底层操作
// ============================================================

/// row_swap: 交换两行的全部内容和长度
template<typename Sol>
__device__ inline void row_swap(Sol& sol, int r1, int r2) {
    // 交换长度
    int tmp_size = sol.dim2_sizes[r1];
    sol.dim2_sizes[r1] = sol.dim2_sizes[r2];
    sol.dim2_sizes[r2] = tmp_size;
    // 交换数据（取两行中较长的长度）
    int max_len = (sol.dim2_sizes[r1] > sol.dim2_sizes[r2]) 
                  ? sol.dim2_sizes[r1] : sol.dim2_sizes[r2];
    // 交换后 r1 的长度是原 r2 的，r2 的长度是原 r1 的
    // 所以需要交换 max(原r1长度, 原r2长度) 个元素
    max_len = (tmp_size > max_len) ? tmp_size : max_len;
    for (int c = 0; c < max_len; c++) {
        int tmp = sol.data[r1][c];
        sol.data[r1][c] = sol.data[r2][c];
        sol.data[r2][c] = tmp;
    }
}

/// row_reverse: 反转 [r1, r2] 范围内的行排列顺序
/// 例如 row_reverse(sol, 1, 4) 把行 1,2,3,4 变成 4,3,2,1
template<typename Sol>
__device__ inline void row_reverse_range(Sol& sol, int r1, int r2) {
    while (r1 < r2) {
        row_swap(sol, r1, r2);
        r1++; r2--;
    }
}

/// row_split: 将 row 从 split_pos 处拆成两行
/// row 保留 [0, split_pos)，empty_row 接收 [split_pos, size)
/// 要求 empty_row 当前为空或有足够空间
template<typename Sol>
__device__ inline void row_split(Sol& sol, int row, int empty_row, int split_pos) {
    int orig_size = sol.dim2_sizes[row];
    int move_count = orig_size - split_pos;
    // 复制后半段到 empty_row
    for (int i = 0; i < move_count; i++)
        sol.data[empty_row][i] = sol.data[row][split_pos + i];
    sol.dim2_sizes[empty_row] = move_count;
    sol.dim2_sizes[row] = split_pos;
}

/// row_merge: 将 src_row 的全部内容追加到 dst_row 末尾
/// src_row 清空，dst_row 长度增加
/// 要求 dst_size + src_size <= DIM2
template<typename Sol>
__device__ inline void row_merge(Sol& sol, int dst_row, int src_row) {
    int dst_size = sol.dim2_sizes[dst_row];
    int src_size = sol.dim2_sizes[src_row];
    for (int i = 0; i < src_size; i++)
        sol.data[dst_row][dst_size + i] = sol.data[src_row][i];
    sol.dim2_sizes[dst_row] = dst_size + src_size;
    sol.dim2_sizes[src_row] = 0;
}

// ============================================================
// 第 4 层：交叉底层操作
// ============================================================
//
// 排列编码：OX 家族（统一框架）
//   核心逻辑：A 中标记一组"保留位置"不动，空位按 B 的全局顺序填充
//   三个变体只是"怎么选保留集"不同，填充逻辑完全共享
//   天然保证唯一性：从 B 中按序取不在保留集中的元素，不会重复
//   行长度不变（= A 的行长度），行边界不变
//
// Binary 编码：uniform_crossover（逐元素随机选）
//
// ============================================================

// ---- OX 核心填充逻辑 ----
// keep[r][c] = true 表示 child[r][c] 保留 A 的值，false 表示空位
// 空位按 B 中元素的出现顺序（逐行扫描）填充
// 要求：child 已拷贝自 A，dim2_sizes 已设为 A 的行长度
//
// 参数 total_elements: 分区模式下的总元素数，非分区模式下 = 单行长度
//   用于确定 B 中扫描的元素范围

template<typename Sol>
__device__ inline void ox_fill_from_b(Sol& child, const Sol& parentB,
                                       const bool* keep_flat,
                                       int dim1, int total_elements) {
    // 统计 A 中保留位置的每个值的出现次数（支持多重集排列）
    // keep_flat 是按行展平的：keep_flat[r * DIM2 + c]
    int keep_count[512];
    for (int i = 0; i < total_elements; i++) keep_count[i] = 0;
    
    for (int r = 0; r < dim1; r++)
        for (int c = 0; c < child.dim2_sizes[r]; c++)
            if (keep_flat[r * Sol::DIM2 + c]) {
                int v = child.data[r][c];
                if (v >= 0 && v < total_elements) keep_count[v]++;
            }
    
    // 从 B 中按行扫描顺序收集：每个值只取"需要填充"的份数
    // 标准排列：每个值最多 1 份，多重集：每个值最多 repeat_count 份
    int fill_buf[512];
    int fill_count = 0;
    for (int r = 0; r < dim1; r++)
        for (int c = 0; c < parentB.dim2_sizes[r]; c++) {
            int val = parentB.data[r][c];
            if (val >= 0 && val < total_elements && keep_count[val] > 0) {
                keep_count[val]--;  // 消耗一个保留名额
            } else if (val >= 0 && val < total_elements) {
                fill_buf[fill_count++] = val;
            }
        }
    
    // 按空位顺序（逐行从左到右）填入
    int fi = 0;
    for (int r = 0; r < dim1; r++)
        for (int c = 0; c < child.dim2_sizes[r]; c++)
            if (!keep_flat[r * Sol::DIM2 + c] && fi < fill_count)
                child.data[r][c] = fill_buf[fi++];
}

// ---- 变体 1: OX-区间 ----
// 每行随机选一个连续区间保留，保留邻接关系
template<typename Sol>
__device__ inline void ox_interval(Sol& child, const Sol& parentA, const Sol& parentB,
                                    int dim1, int total_elements, curandState* rng) {
    bool keep[Sol::DIM1 * Sol::DIM2];
    for (int i = 0; i < Sol::DIM1 * Sol::DIM2; i++) keep[i] = false;
    
    // child = A，同时标记每行的保留区间
    for (int r = 0; r < dim1; r++) {
        int sz = parentA.dim2_sizes[r];
        child.dim2_sizes[r] = sz;
        for (int c = 0; c < sz; c++) child.data[r][c] = parentA.data[r][c];
        
        if (sz < 2) {
            // 长度 0 或 1：全部保留
            for (int c = 0; c < sz; c++) keep[r * Sol::DIM2 + c] = true;
            continue;
        }
        // 随机选区间 [lo, hi]
        int lo = rand_int(rng, sz);
        int hi = rand_int(rng, sz);
        if (lo > hi) { int tmp = lo; lo = hi; hi = tmp; }
        for (int c = lo; c <= hi; c++) keep[r * Sol::DIM2 + c] = true;
    }
    
    ox_fill_from_b(child, parentB, keep, dim1, total_elements);
}

// ---- 变体 2: OX-子集 ----
// 随机选约 50% 的元素值保留其在 A 中的位置，通用性最强
template<typename Sol>
__device__ inline void ox_subset(Sol& child, const Sol& parentA, const Sol& parentB,
                                  int dim1, int total_elements, curandState* rng) {
    bool keep[Sol::DIM1 * Sol::DIM2];
    for (int i = 0; i < Sol::DIM1 * Sol::DIM2; i++) keep[i] = false;
    
    // child = A
    for (int r = 0; r < dim1; r++) {
        child.dim2_sizes[r] = parentA.dim2_sizes[r];
        for (int c = 0; c < parentA.dim2_sizes[r]; c++)
            child.data[r][c] = parentA.data[r][c];
    }
    
    // 每个位置 50% 概率保留
    for (int r = 0; r < dim1; r++)
        for (int c = 0; c < child.dim2_sizes[r]; c++)
            keep[r * Sol::DIM2 + c] = (curand_uniform(rng) < 0.5f);
    
    ox_fill_from_b(child, parentB, keep, dim1, total_elements);
}

// ---- 变体 3: OX-行 ----
// 随机选若干整行保留，其余行的元素全部按 B 的顺序重填
// 保留整条路线结构，VRP 受益
template<typename Sol>
__device__ inline void ox_row(Sol& child, const Sol& parentA, const Sol& parentB,
                               int dim1, int total_elements, curandState* rng) {
    bool keep[Sol::DIM1 * Sol::DIM2];
    for (int i = 0; i < Sol::DIM1 * Sol::DIM2; i++) keep[i] = false;
    
    // child = A
    for (int r = 0; r < dim1; r++) {
        child.dim2_sizes[r] = parentA.dim2_sizes[r];
        for (int c = 0; c < parentA.dim2_sizes[r]; c++)
            child.data[r][c] = parentA.data[r][c];
    }
    
    // 每行 50% 概率整行保留
    int kept = 0;
    for (int r = 0; r < dim1; r++) {
        if (curand_uniform(rng) < 0.5f) {
            for (int c = 0; c < child.dim2_sizes[r]; c++)
                keep[r * Sol::DIM2 + c] = true;
            kept++;
        }
    }
    // 确保不是全保留或全不保留
    if (kept == 0) {
        int r = rand_int(rng, dim1);
        // 不标记任何 keep → 全部重填（至少有一行不保留）
        // 实际上 kept==0 意味着全部重填，这是合法的（child = B 的顺序填入 A 的结构）
    }
    if (kept == dim1 && dim1 > 1) {
        // 全保留 → 随机取消一行
        int r = rand_int(rng, dim1);
        for (int c = 0; c < child.dim2_sizes[r]; c++)
            keep[r * Sol::DIM2 + c] = false;
    }
    
    ox_fill_from_b(child, parentB, keep, dim1, total_elements);
}

// ---- OX 统一入口 ----
// 随机选一个变体执行
// dim1==1 时只用区间和子集（行变体无意义）
template<typename Sol>
__device__ inline void perm_ox_crossover(Sol& child, const Sol& parentA, const Sol& parentB,
                                          int dim1, int total_elements, curandState* rng) {
    int n_variants = (dim1 > 1) ? 3 : 2;
    int variant = rand_int(rng, n_variants);  // 0: 区间, 1: 子集, [2: 行]
    switch (variant) {
        case 0: ox_interval(child, parentA, parentB, dim1, total_elements, rng); break;
        case 1: ox_subset(child, parentA, parentB, dim1, total_elements, rng); break;
        case 2: ox_row(child, parentA, parentB, dim1, total_elements, rng); break;
    }
}

/// uniform_crossover: 逐元素从两个父代中随机选择
/// 适用于 Binary 编码（不破坏排列约束）
template<typename Sol>
__device__ inline void uniform_crossover(Sol& child, const Sol& parentA, const Sol& parentB,
                                          int dim1, curandState* rng) {
    for (int r = 0; r < dim1; r++) {
        int sizeA = parentA.dim2_sizes[r];
        int sizeB = parentB.dim2_sizes[r];
        int size = (sizeA < sizeB) ? sizeA : sizeB;
        child.dim2_sizes[r] = size;
        for (int c = 0; c < size; c++) {
            child.data[r][c] = (curand_uniform(rng) < 0.5f)
                               ? parentA.data[r][c] : parentB.data[r][c];
        }
    }
}

// [已删除] generate_move_for_seq / sample_and_generate / apply_move / undo_move
// P0 重构后主路径统一使用 execute_sequence，旧的 Move 生成+应用+撤销路径不再需要

// ============================================================
// execute_sequence — 统一接口：生成参数并直接执行（不返回 Move）
// ============================================================
// 返回 true 若 sol 被修改，false 若 NOOP
// d_G, d_O, rel_N: 可选的关系矩阵指针（SEQ_LNS_GUIDED_REBUILD 使用）
// val_lb, val_ub: Integer 编码的值域范围（其他编码忽略）

template<typename Sol>
__device__ inline bool execute_sequence(int seq_id, Sol& sol, int dim1,
                                         EncodingType encoding, curandState* rng,
                                         const float* d_G = nullptr,
                                         const float* d_O = nullptr,
                                         int rel_N = 0,
                                         int val_lb = 0,
                                         int val_ub = 1,
                                         const void* prob_data = nullptr) {
    // ============================================================
    // Permutation 序列
    // ============================================================
    if (encoding == EncodingType::Permutation) {
        switch (seq_id) {
        case seq::SEQ_PERM_SWAP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int pos1 = rand_int(rng, sz);
            int pos2 = rand_int(rng, sz - 1); if (pos2 >= pos1) pos2++;
            perm_swap(sol.data[row], pos1, pos2);
            return true;
        }
        case seq::SEQ_PERM_REVERSE: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int pos1 = rand_int(rng, sz);
            int pos2 = rand_int(rng, sz - 1); if (pos2 >= pos1) pos2++;
            if (pos1 > pos2) { int t = pos1; pos1 = pos2; pos2 = t; }
            perm_reverse(sol.data[row], pos1, pos2);
            return true;
        }
        case seq::SEQ_PERM_INSERT: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int pos1 = rand_int(rng, sz);
            int pos2 = rand_int(rng, sz - 1); if (pos2 >= pos1) pos2++;
            perm_insert(sol.data[row], pos1, pos2, sz);
            return true;
        }
        case seq::SEQ_PERM_3OPT: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 5) return false;
            int a = rand_int(rng, sz);
            int b = rand_int(rng, sz - 1); if (b >= a) b++;
            int c = rand_int(rng, sz - 2); if (c >= a) c++; if (c >= b) c++;
            if (a > b) { int t = a; a = b; b = t; }
            if (b > c) { int t = b; b = c; c = t; }
            if (a > b) { int t = a; a = b; b = t; }
            perm_3opt(sol.data[row], sz, a, b, c);
            return true;
        }
        case seq::SEQ_PERM_OR_OPT: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 3) return false;
            int max_seg = (sz < 4) ? sz - 1 : 4;
            int seg_len = 1 + rand_int(rng, max_seg);
            int pos1 = rand_int(rng, sz - seg_len + 1);
            int avail = sz - seg_len;
            if (avail < 1) return false;
            int pos2 = rand_int(rng, avail);
            if (pos2 >= pos1) pos2 += seg_len;
            perm_or_opt(sol.data[row], sz, pos1, pos2, seg_len);
            return true;
        }
        case seq::SEQ_PERM_CROSS_RELOCATE: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz == 0 && dst_sz > 0) {
                int t = row; row = row2; row2 = t;
                src_sz = sol.dim2_sizes[row]; dst_sz = sol.dim2_sizes[row2];
            }
            if (src_sz == 0 || dst_sz >= Sol::DIM2) return false;
            int pos1 = rand_int(rng, src_sz);
            int pos2 = rand_int(rng, dst_sz + 1);
            perm_cross_relocate(sol.data[row], sol.dim2_sizes[row],
                               sol.data[row2], sol.dim2_sizes[row2],
                               pos1, pos2);
            return true;
        }
        case seq::SEQ_PERM_CROSS_SWAP: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz == 0 || dst_sz == 0) return false;
            int pos1 = rand_int(rng, src_sz);
            int pos2 = rand_int(rng, dst_sz);
            cross_swap_elem(sol.data[row], pos1, sol.data[row2], pos2);
            return true;
        }
        case seq::SEQ_PERM_SEG_RELOCATE: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz == 0 && dst_sz > 0) {
                int t = row; row = row2; row2 = t;
                src_sz = sol.dim2_sizes[row]; dst_sz = sol.dim2_sizes[row2];
            }
            if (src_sz < 2) return false;
            int max_seg = (src_sz < 4) ? src_sz : 4;
            int seg_len = 2 + rand_int(rng, max_seg - 1);
            if (seg_len > src_sz) seg_len = src_sz;
            if (dst_sz + seg_len > Sol::DIM2) return false;
            int pos1 = rand_int(rng, src_sz - seg_len + 1);
            int pos2 = rand_int(rng, dst_sz + 1);
            perm_seg_relocate(sol.data[row], sol.dim2_sizes[row],
                             sol.data[row2], sol.dim2_sizes[row2],
                             pos1, pos2, seg_len);
            return true;
        }
        case seq::SEQ_PERM_SEG_SWAP: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz < 2 || dst_sz < 2) return false;
            int max_seg = (src_sz < 4) ? src_sz : 4;
            int seg_len = 2 + rand_int(rng, max_seg - 1);
            if (seg_len > src_sz) seg_len = src_sz;
            if (dst_sz < seg_len) return false;
            int pos1 = rand_int(rng, src_sz - seg_len + 1);
            int avail = dst_sz - seg_len + 1;
            if (avail <= 0) return false;
            int pos2 = rand_int(rng, avail);
            perm_seg_swap(sol.data[row], sol.dim2_sizes[row], pos1, seg_len,
                         sol.data[row2], sol.dim2_sizes[row2], pos2, seg_len);
            return true;
        }
        case seq::SEQ_PERM_CROSS_EXCHANGE: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz < 2 || dst_sz < 2) return false;
            int max_segA = (src_sz < 4) ? src_sz : 4;
            int lenA = 2 + rand_int(rng, max_segA - 1);
            if (lenA > src_sz) lenA = src_sz;
            int dst_max_seg = (dst_sz < 4) ? dst_sz : 4;
            int lenB = 1 + rand_int(rng, dst_max_seg);
            if (src_sz - lenA + lenB > Sol::DIM2 || dst_sz - lenB + lenA > Sol::DIM2)
                return false;
            int pos1 = rand_int(rng, src_sz - lenA + 1);
            int pos2 = rand_int(rng, dst_sz - lenB + 1);
            perm_cross_exchange(sol.data[row], sol.dim2_sizes[row], pos1, lenA,
                               sol.data[row2], sol.dim2_sizes[row2], pos2, lenB);
            return true;
        }
        case seq::SEQ_PERM_DOUBLE_SWAP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 3) return false;
            int a1 = rand_int(rng, sz), a2 = rand_int(rng, sz - 1); if (a2 >= a1) a2++;
            perm_swap(sol.data[row], a1, a2);
            int b1 = rand_int(rng, sz), b2 = rand_int(rng, sz - 1); if (b2 >= b1) b2++;
            perm_swap(sol.data[row], b1, b2);
            return true;
        }
        case seq::SEQ_PERM_TRIPLE_SWAP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 4) return false;
            for (int s = 0; s < 3; s++) {
                int p1 = rand_int(rng, sz), p2 = rand_int(rng, sz - 1); if (p2 >= p1) p2++;
                perm_swap(sol.data[row], p1, p2);
            }
            return true;
        }
        case seq::SEQ_LNS_SEGMENT_SHUFFLE: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 4) return false;
            int count = sz / 5;
            if (count < 2) count = 2;
            if (count > sz) count = sz;
            int start = rand_int(rng, sz);
            for (int i = count - 1; i > 0; i--) {
                int j = rand_int(rng, i + 1);
                int pi = (start + i) % sz;
                int pj = (start + j) % sz;
                int tmp = sol.data[row][pi];
                sol.data[row][pi] = sol.data[row][pj];
                sol.data[row][pj] = tmp;
            }
            return true;
        }
        case seq::SEQ_LNS_SCATTER_SHUFFLE: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 4) return false;
            int count = sz / 5;
            if (count < 2) count = 2;
            if (count > 16) count = 16;
            int positions[16];
            for (int i = 0; i < count; i++) {
                positions[i] = rand_int(rng, sz);
                for (int j = 0; j < i; j++) {
                    if (positions[i] == positions[j]) {
                        positions[i] = rand_int(rng, sz);
                        j = -1;
                    }
                }
            }
            for (int i = count - 1; i > 0; i--) {
                int j = rand_int(rng, i + 1);
                int tmp = sol.data[row][positions[i]];
                sol.data[row][positions[i]] = sol.data[row][positions[j]];
                sol.data[row][positions[j]] = tmp;
            }
            return true;
        }
        case seq::SEQ_LNS_GUIDED_REBUILD: {
            // 关系矩阵引导重建：
            //   1. 随机选种子元素 seed
            //   2. 查 G[seed] 找分组倾向最强的 K 个元素
            //   3. 在解中找到这些元素的位置
            //   4. 按 O 矩阵引导的顺序重排这些位置的元素
            //
            // 如果没有关系矩阵（冷启动），退化为 scatter_shuffle
            if (!d_G || !d_O || rel_N <= 0) {
                // 退化：随机 scatter shuffle
                int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
                int sz = sol.dim2_sizes[row];
                if (sz < 4) return false;
                int count = sz / 5;
                if (count < 2) count = 2;
                if (count > 12) count = 12;
                int positions[12];
                for (int i = 0; i < count; i++) {
                    positions[i] = rand_int(rng, sz);
                    for (int j = 0; j < i; j++) {
                        if (positions[i] == positions[j]) { positions[i] = rand_int(rng, sz); j = -1; }
                    }
                }
                for (int i = count - 1; i > 0; i--) {
                    int j = rand_int(rng, i + 1);
                    int tmp = sol.data[row][positions[i]];
                    sol.data[row][positions[i]] = sol.data[row][positions[j]];
                    sol.data[row][positions[j]] = tmp;
                }
                return true;
            }
            
            // --- 有关系矩阵：引导重建 ---
            // 通用策略（不感知问题类型）：
            //   G 矩阵 → 选哪些元素（分组倾向弱的 = 可能放错位置的）
            //   O 矩阵 → 怎么排（排序倾向引导重排顺序）
            //   两者协同：G 选人，O 排序
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 4) return false;
            
            // 选种子元素
            int seed_pos = rand_int(rng, sz);
            int seed_val = sol.data[row][seed_pos];
            if (seed_val < 0 || seed_val >= rel_N) return false;
            
            // 检查矩阵是否有足够信息（G 和 O 任一有信号即可）
            float max_signal = 0.0f;
            for (int c = 0; c < sz; c++) {
                int v = sol.data[row][c];
                if (v >= 0 && v < rel_N && v != seed_val) {
                    float g = d_G[seed_val * rel_N + v];
                    float o = d_O[seed_val * rel_N + v];
                    if (g > max_signal) max_signal = g;
                    if (o > max_signal) max_signal = o;
                }
            }
            if (max_signal < 0.05f) return false;  // 信息不足，跳过
            
            // 破坏：锦标赛选择 G 值低的元素（t=2）
            // G 值低 = 与 seed 分组倾向弱 = 可能放错位置
            // 锦标赛：随机抽 2 个，取 G 值更低的那个，重复 count 次
            constexpr int MAX_REBUILD = 10;
            constexpr int TOUR_SIZE = 2;
            int count = sz / 5;  // ~20%
            if (count < 3) count = 3;
            if (count > MAX_REBUILD) count = MAX_REBUILD;
            if (count > sz) count = sz;
            
            int sel_pos[MAX_REBUILD];
            int sel_val[MAX_REBUILD];
            bool used[128] = {};  // 标记已选位置，防止重复
            int picked = 0;
            int max_attempts = count * 4;  // 防止死循环
            
            for (int attempt = 0; attempt < max_attempts && picked < count; attempt++) {
                // 锦标赛：随机抽 TOUR_SIZE 个候选，取 G 值最低的
                int best_c = -1;
                float best_g = 1e30f;
                for (int t = 0; t < TOUR_SIZE; t++) {
                    int c = rand_int(rng, sz);
                    if (used[c]) continue;
                    int v = sol.data[row][c];
                    if (v < 0 || v >= rel_N) continue;
                    float g = d_G[seed_val * rel_N + v];
                    if (g < best_g) { best_g = g; best_c = c; }
                }
                if (best_c < 0 || used[best_c]) continue;
                used[best_c] = true;
                sel_pos[picked] = best_c;
                sel_val[picked] = sol.data[row][best_c];
                picked++;
            }
            if (picked < 2) return false;
            count = picked;
            
            // 修复：锦标赛排序（O 矩阵引导 + 随机扰动）
            // 插入排序，比较时加噪声实现概率性：O 值高的大概率排前面，但不绝对
            for (int i = 1; i < count; i++) {
                int key = sel_val[i];
                int j = i - 1;
                while (j >= 0) {
                    float o_key_before = d_O[key * rel_N + sel_val[j]];
                    float o_j_before   = d_O[sel_val[j] * rel_N + key];
                    // 噪声幅度 0.05：O 值差距 >0.05 时基本确定，<0.05 时随机
                    float noise = (curand_uniform(rng) - 0.5f) * 0.1f;
                    if (o_key_before + noise > o_j_before) {
                        sel_val[j + 1] = sel_val[j];
                        j--;
                    } else {
                        break;
                    }
                }
                sel_val[j + 1] = key;
            }
            
            // 对 sel_pos 排序（升序），使写回位置有序
            for (int i = 1; i < count; i++) {
                int key = sel_pos[i];
                int j = i - 1;
                while (j >= 0 && sel_pos[j] > key) {
                    sel_pos[j + 1] = sel_pos[j];
                    j--;
                }
                sel_pos[j + 1] = key;
            }
            
            // 检查是否真的改变了排列
            bool any_change = false;
            for (int i = 0; i < count; i++) {
                if (sol.data[row][sel_pos[i]] != sel_val[i]) {
                    any_change = true;
                    break;
                }
            }
            if (!any_change) return false;
            
            // 写回
            for (int i = 0; i < count; i++) {
                sol.data[row][sel_pos[i]] = sel_val[i];
            }
            
            return true;
        }
        default: break;
        }
    }

    // ============================================================
    // Binary 序列
    // ============================================================
    if (encoding == EncodingType::Binary) {
        switch (seq_id) {
        case seq::SEQ_BIN_FLIP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 1) return false;
            int pos1 = rand_int(rng, sz);
            bin_flip(sol.data[row], pos1);
            return true;
        }
        case seq::SEQ_BIN_SWAP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int pos1 = rand_int(rng, sz);
            int pos2 = rand_int(rng, sz - 1); if (pos2 >= pos1) pos2++;
            bin_swap(sol.data[row], pos1, pos2);
            return true;
        }
        case seq::SEQ_BIN_SEG_FLIP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int max_seg = (sz < 4) ? sz : 4;
            int seg_len = 2 + rand_int(rng, max_seg - 1);
            if (seg_len > sz) seg_len = sz;
            int pos1 = rand_int(rng, sz - seg_len + 1);
            bin_seg_flip(sol.data[row], pos1, seg_len);
            return true;
        }
        case seq::SEQ_BIN_K_FLIP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int max_k = (sz < 5) ? sz : 5;
            int k = 2 + rand_int(rng, max_k - 1);
            bin_k_flip(sol.data[row], sz, k, rng);
            return true;
        }
        case seq::SEQ_BIN_CROSS_SWAP: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz == 0 || dst_sz == 0) return false;
            int pos1 = rand_int(rng, src_sz);
            int pos2 = rand_int(rng, dst_sz);
            cross_swap_elem(sol.data[row], pos1, sol.data[row2], pos2);
            return true;
        }
        case seq::SEQ_BIN_SEG_CROSS_SWAP: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz < 2 || dst_sz < 2) return false;
            int max_seg = (src_sz < 4) ? src_sz : 4;
            int seg_len = 2 + rand_int(rng, max_seg - 1);
            if (seg_len > src_sz) seg_len = src_sz;
            if (dst_sz < seg_len) return false;
            int pos1 = rand_int(rng, src_sz - seg_len + 1);
            int pos2 = rand_int(rng, dst_sz - seg_len + 1);
            bin_seg_cross_swap(sol.data[row], pos1, sol.data[row2], pos2, seg_len);
            return true;
        }
        default: break;
        }
    }

    // ============================================================
    // Integer 序列
    // ============================================================
    if (encoding == EncodingType::Integer) {
        switch (seq_id) {
        case seq::SEQ_INT_RANDOM_RESET: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 1) return false;
            int pos = rand_int(rng, sz);
            int_random_reset(sol.data[row], pos, val_lb, val_ub, rng);
            return true;
        }
        case seq::SEQ_INT_DELTA: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 1) return false;
            int pos = rand_int(rng, sz);
            int_delta(sol.data[row], pos, val_lb, val_ub, rng);
            return true;
        }
        case seq::SEQ_INT_SWAP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int pos1 = rand_int(rng, sz);
            int pos2 = rand_int(rng, sz - 1); if (pos2 >= pos1) pos2++;
            int tmp = sol.data[row][pos1];
            sol.data[row][pos1] = sol.data[row][pos2];
            sol.data[row][pos2] = tmp;
            return true;
        }
        case seq::SEQ_INT_SEG_RESET: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int max_seg = (sz < 4) ? sz : 4;
            int seg_len = 2 + rand_int(rng, max_seg - 1);
            if (seg_len > sz) seg_len = sz;
            int pos = rand_int(rng, sz - seg_len + 1);
            int_seg_reset(sol.data[row], pos, seg_len, val_lb, val_ub, rng);
            return true;
        }
        case seq::SEQ_INT_K_DELTA: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int max_k = (sz < 5) ? sz : 5;
            int k = 2 + rand_int(rng, max_k - 1);
            int_k_delta(sol.data[row], sz, k, val_lb, val_ub, rng);
            return true;
        }
        case seq::SEQ_INT_CROSS_SWAP: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz == 0 || dst_sz == 0) return false;
            int pos1 = rand_int(rng, src_sz);
            int pos2 = rand_int(rng, dst_sz);
            cross_swap_elem(sol.data[row], pos1, sol.data[row2], pos2);
            return true;
        }
        default: break;
        }
    }

    // ============================================================
    // 共享：行级序列（编码无关）
    // ============================================================
    switch (seq_id) {
    case seq::SEQ_ROW_SWAP: {
        if (dim1 < 2) return false;
        int r1 = rand_int(rng, dim1);
        int r2 = rand_int(rng, dim1 - 1); if (r2 >= r1) r2++;
        row_swap(sol, r1, r2);
        return true;
    }
    case seq::SEQ_ROW_REVERSE: {
        if (dim1 < 2) return false;
        int r1 = rand_int(rng, dim1);
        int r2 = rand_int(rng, dim1 - 1); if (r2 >= r1) r2++;
        if (r1 > r2) { int t = r1; r1 = r2; r2 = t; }
        row_reverse_range(sol, r1, r2);
        return true;
    }
    case seq::SEQ_ROW_SPLIT: {
        if (dim1 < 2) return false;
        int attempts = 0;
        int row;
        do { row = rand_int(rng, dim1); attempts++; }
        while (sol.dim2_sizes[row] < 2 && attempts < dim1 * 2);
        if (sol.dim2_sizes[row] < 2) return false;
        int empty_row = -1;
        for (int r = 0; r < dim1; r++) {
            if (r != row && sol.dim2_sizes[r] == 0) { empty_row = r; break; }
        }
        if (empty_row < 0) return false;
        int split_pos = 1 + rand_int(rng, sol.dim2_sizes[row] - 1);
        row_split(sol, row, empty_row, split_pos);
        return true;
    }
    case seq::SEQ_ROW_MERGE: {
        if (dim1 < 2) return false;
        int non_empty[64]; int cnt = 0;
        for (int r = 0; r < dim1 && cnt < 64; r++)
            if (sol.dim2_sizes[r] > 0) non_empty[cnt++] = r;
        if (cnt < 2) return false;
        int i1 = rand_int(rng, cnt);
        int i2 = rand_int(rng, cnt - 1); if (i2 >= i1) i2++;
        int dst_row = non_empty[i1];
        int src_row = non_empty[i2];
        if (sol.dim2_sizes[dst_row] + sol.dim2_sizes[src_row] > Sol::DIM2)
            return false;
        row_merge(sol, dst_row, src_row);
        return true;
    }
    default:
        break;
    }

    // Custom operator hook: if seq_id >= 100, delegate to user-defined function
    // (defined via JIT template or left as default no-op)
#ifdef CUGENOPT_HAS_CUSTOM_OPS
    return execute_custom_op(seq_id, sol, dim1, encoding, rng, val_lb, val_ub, prob_data);
#else
    return false;
#endif
}

// ============================================================
// sample_and_execute — 从 SeqRegistry 按权重采样 + 直接执行
// ============================================================
// 返回 true 若 sol 被修改，false 若 NOOP
// 输出参数 out_seq_idx：采样到的序列在 registry 中的索引
// d_G, d_O, rel_N: 可选的关系矩阵（传递给 execute_sequence）

template<typename Sol>
__device__ inline bool sample_and_execute(const SeqRegistry& reg,
                                          Sol& sol, int dim1,
                                          EncodingType encoding,
                                          curandState* rng,
                                          int& out_seq_idx,
                                          const float* d_G = nullptr,
                                          const float* d_O = nullptr,
                                          int rel_N = 0,
                                          int val_lb = 0,
                                          int val_ub = 1,
                                          const void* prob_data = nullptr) {
    // 延迟归一化：使用缓存的 weights_sum
    float r = curand_uniform(rng) * reg.weights_sum;  // r ∈ [0, weights_sum)
    float cumsum = 0.0f;
    out_seq_idx = reg.count - 1;
    for (int i = 0; i < reg.count; i++) {
        cumsum += reg.weights[i];
        if (r < cumsum) { out_seq_idx = i; break; }
    }
    int seq_id = reg.ids[out_seq_idx];
    return execute_sequence(seq_id, sol, dim1, encoding, rng, d_G, d_O, rel_N,
                            val_lb, val_ub, prob_data);
}



}  // namespace ops
