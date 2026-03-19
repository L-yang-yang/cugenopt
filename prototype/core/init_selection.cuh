/**
 * init_selection.cuh - 初始解采样择优 + NSGA-II 选择
 *
 * Host 端逻辑，在 solver 初始化阶段调用一次。
 * 从 K × pop_size 个候选解中选出 pop_size 个作为初始种群。
 *
 * 选择策略：
 *   1. 核心目标预留名额（按 importance 分配）
 *   2. NSGA-II 选择（非支配排序 + 加权拥挤度）
 *   3. 纯随机保底（多样性）
 *
 * 单目标时自动退化为 top-N 排序，无需分支。
 */

#pragma once
#include "types.cuh"
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>

namespace init_sel {

// ============================================================
// 候选解的目标信息（从 GPU 下载后在 host 端使用）
// ============================================================
struct CandidateInfo {
    int   idx;           // 在候选数组中的原始索引
    float objs[MAX_OBJ]; // 归一化后的目标值（越小越好）
    float penalty;
    int   rank;          // 非支配排序层级（0 = Pareto 前沿）
    float crowding;      // 拥挤度距离
    bool  selected;      // 是否已被选中
};

// ============================================================
// 非支配排序（Fast Non-dominated Sort）
// ============================================================
// 复杂度：O(M × N²)，M = 目标数，N = 候选数
// 对初始化场景（N ≤ 几千，M ≤ 4）完全可接受

inline void fast_nondominated_sort(std::vector<CandidateInfo>& cands,
                                    int num_obj,
                                    std::vector<std::vector<int>>& fronts) {
    int n = (int)cands.size();
    std::vector<int> dom_count(n, 0);        // 被多少个解支配
    std::vector<std::vector<int>> dom_set(n); // 支配了哪些解
    
    // 判断 a 是否支配 b：a 在所有目标上 ≤ b，且至少一个 <
    // 先处理 penalty：可行解支配不可行解
    auto dominates = [&](int a, int b) -> bool {
        const auto& ca = cands[a];
        const auto& cb = cands[b];
        // penalty 处理
        if (ca.penalty <= 0.0f && cb.penalty > 0.0f) return true;
        if (ca.penalty > 0.0f && cb.penalty <= 0.0f) return false;
        if (ca.penalty > 0.0f && cb.penalty > 0.0f) return ca.penalty < cb.penalty;
        
        bool all_leq = true;
        bool any_lt = false;
        for (int m = 0; m < num_obj; m++) {
            if (ca.objs[m] > cb.objs[m]) { all_leq = false; break; }
            if (ca.objs[m] < cb.objs[m]) any_lt = true;
        }
        return all_leq && any_lt;
    };
    
    // 计算支配关系
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (dominates(i, j)) {
                dom_set[i].push_back(j);
                dom_count[j]++;
            } else if (dominates(j, i)) {
                dom_set[j].push_back(i);
                dom_count[i]++;
            }
        }
    }
    
    // 提取各层前沿
    fronts.clear();
    std::vector<int> current_front;
    for (int i = 0; i < n; i++) {
        if (dom_count[i] == 0) {
            cands[i].rank = 0;
            current_front.push_back(i);
        }
    }
    
    int front_idx = 0;
    while (!current_front.empty()) {
        fronts.push_back(current_front);
        std::vector<int> next_front;
        for (int i : current_front) {
            for (int j : dom_set[i]) {
                dom_count[j]--;
                if (dom_count[j] == 0) {
                    cands[j].rank = front_idx + 1;
                    next_front.push_back(j);
                }
            }
        }
        current_front = next_front;
        front_idx++;
    }
}

// ============================================================
// 加权拥挤度距离
// ============================================================
// 标准拥挤度 + importance 加权：核心目标维度上的间距贡献更大

inline void weighted_crowding_distance(std::vector<CandidateInfo>& cands,
                                        const std::vector<int>& front,
                                        int num_obj,
                                        const float* importance) {
    int n = (int)front.size();
    if (n <= 2) {
        for (int i : front) cands[i].crowding = 1e18f;  // 边界解无穷大
        return;
    }
    
    for (int i : front) cands[i].crowding = 0.0f;
    
    std::vector<int> sorted_idx(front.begin(), front.end());
    
    for (int m = 0; m < num_obj; m++) {
        // 按目标 m 排序
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&](int a, int b) { return cands[a].objs[m] < cands[b].objs[m]; });
        
        float range = cands[sorted_idx[n-1]].objs[m] - cands[sorted_idx[0]].objs[m];
        if (range < 1e-12f) continue;  // 该目标无区分度
        
        // 边界解设为无穷大
        cands[sorted_idx[0]].crowding += 1e18f;
        cands[sorted_idx[n-1]].crowding += 1e18f;
        
        // 中间解：相邻间距 × importance 权重
        float w = importance[m];
        for (int i = 1; i < n - 1; i++) {
            float gap = cands[sorted_idx[i+1]].objs[m] - cands[sorted_idx[i-1]].objs[m];
            cands[sorted_idx[i]].crowding += w * (gap / range);
        }
    }
}

// ============================================================
// 主选择函数：从 N 个候选中选出 target 个
// ============================================================
// 返回被选中的候选索引

inline std::vector<int> nsga2_select(std::vector<CandidateInfo>& cands,
                                      int num_obj,
                                      const float* importance,
                                      int target,
                                      int num_reserved_random) {
    // --- 1. 核心目标预留名额 ---
    int num_reserve_total = target - num_reserved_random;
    // 预留比例：importance[i] × 30% 的名额（剩余 70% 给 NSGA-II）
    float reserve_ratio = 0.3f;
    
    std::vector<int> selected;
    selected.reserve(target);
    
    // 对每个目标，按该目标排序取 top
    for (int m = 0; m < num_obj; m++) {
        int quota = (int)(num_reserve_total * importance[m] * reserve_ratio);
        if (quota < 1 && num_obj > 1) quota = 1;  // 每个目标至少 1 个
        
        // 按目标 m 排序（越小越好）
        std::vector<int> by_obj(cands.size());
        for (int i = 0; i < (int)cands.size(); i++) by_obj[i] = i;
        std::sort(by_obj.begin(), by_obj.end(),
                  [&](int a, int b) { return cands[a].objs[m] < cands[b].objs[m]; });
        
        int added = 0;
        for (int i = 0; i < (int)by_obj.size() && added < quota; i++) {
            int idx = by_obj[i];
            if (!cands[idx].selected) {
                cands[idx].selected = true;
                selected.push_back(idx);
                added++;
            }
        }
    }
    
    // --- 2. NSGA-II 选择填充剩余名额 ---
    int remaining = target - num_reserved_random - (int)selected.size();
    
    if (remaining > 0) {
        // 非支配排序
        std::vector<std::vector<int>> fronts;
        fast_nondominated_sort(cands, num_obj, fronts);
        
        for (auto& front : fronts) {
            if (remaining <= 0) break;
            
            // 过滤已选中的
            std::vector<int> available;
            for (int i : front) {
                if (!cands[i].selected) available.push_back(i);
            }
            
            if ((int)available.size() <= remaining) {
                // 整层都选
                for (int i : available) {
                    cands[i].selected = true;
                    selected.push_back(i);
                    remaining--;
                }
            } else {
                // 该层需要截断：按加权拥挤度选
                weighted_crowding_distance(cands, available, num_obj, importance);
                std::sort(available.begin(), available.end(),
                          [&](int a, int b) { return cands[a].crowding > cands[b].crowding; });
                for (int i = 0; i < remaining; i++) {
                    cands[available[i]].selected = true;
                    selected.push_back(available[i]);
                }
                remaining = 0;
            }
        }
    }
    
    return selected;
}

// ============================================================
// 单目标快速路径：直接按标量排序取 top
// ============================================================
inline std::vector<int> top_n_select(std::vector<CandidateInfo>& cands,
                                      int target,
                                      int num_reserved_random) {
    int to_select = target - num_reserved_random;
    
    // 按 penalty 优先，然后按 objs[0]（已归一化为越小越好）
    std::vector<int> indices(cands.size());
    for (int i = 0; i < (int)cands.size(); i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        if (cands[a].penalty <= 0.0f && cands[b].penalty > 0.0f) return true;
        if (cands[a].penalty > 0.0f && cands[b].penalty <= 0.0f) return false;
        if (cands[a].penalty > 0.0f && cands[b].penalty > 0.0f)
            return cands[a].penalty < cands[b].penalty;
        return cands[a].objs[0] < cands[b].objs[0];
    });
    
    std::vector<int> selected;
    selected.reserve(to_select);
    for (int i = 0; i < to_select && i < (int)indices.size(); i++) {
        selected.push_back(indices[i]);
        cands[indices[i]].selected = true;
    }
    return selected;
}

} // namespace init_sel
