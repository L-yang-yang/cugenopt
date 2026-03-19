# Encoding Selection & Dimension Guide

## Encoding Types

cuGenOpt supports three encoding types. Choose based on the nature of the decision variables.

### Permutation

**Use when**: each element appears exactly once (ordering/assignment).

| Scenario | RowMode | D1 | D2 | dim2_default | total_elements |
|----------|---------|----|----|-------------|----------------|
| TSP (n cities) | Single | 1 | next_pow2(n) | n | — |
| QAP (n facilities) | Single | 1 | next_pow2(n) | n | — |
| Assignment (n tasks) | Single | 1 | next_pow2(n) | n | — |
| JSP (m machines, j jobs) | Fixed | next_pow2(m) | next_pow2(j) | j | — |
| VRP (k vehicles, n customers) | Partition | next_pow2(k) | max(next_pow2(n/k*2), 64) | 0 | n |
| VRPTW (k vehicles, n customers) | Partition | next_pow2(k) | max(next_pow2(n/k*2), 64) | 0 | n |

**Partition specifics**:
- `dim2_default = 0` tells the framework to distribute elements across rows
- `total_elements = n` is the count of elements to distribute
- `cross_row_prob` controls how often cross-row operators fire (typically 0.2–0.4)
- Elements are customer/job indices `0..n-1`; depot/source is implicit (not in the solution)

### Binary

**Use when**: each position is a yes/no decision.

| Scenario | RowMode | D1 | D2 | dim2_default |
|----------|---------|----|----|-------------|
| 0-1 Knapsack (n items) | Single | 1 | next_pow2(n) | n |
| Scheduling (n shifts) | Single | 1 | next_pow2(n) | n |
| Subset selection (n candidates) | Single | 1 | next_pow2(n) | n |
| Multi-row scheduling (m workers, n shifts) | Fixed | next_pow2(m) | next_pow2(n) | n |

**Solution values**: `sol.data[row][col]` is 0 or 1.

### Integer

**Use when**: each position takes a bounded integer value.

| Scenario | RowMode | D1 | D2 | dim2_default | lower_bound | upper_bound |
|----------|---------|----|----|-------------|-------------|-------------|
| Graph coloring (n nodes, c colors) | Single | 1 | next_pow2(n) | n | 0 | c-1 |
| Load balancing (n tasks, m machines) | Single | 1 | next_pow2(n) | n | 0 | m-1 |
| Multi-machine scheduling | Fixed | next_pow2(m) | next_pow2(j) | j | 0 | max_time |

**Solution values**: `sol.data[row][col]` is in `[value_lower_bound, value_upper_bound]`.

Set bounds in config:
```cuda
cfg.value_lower_bound = 0;
cfg.value_upper_bound = num_colors - 1;
```

## Dimension Calculation Rules

### D1 and D2 (Template Parameters)

These are **compile-time constants** and define the maximum capacity:
- Must be sufficient for the largest instance you plan to solve
- Power of 2 is recommended for memory alignment
- Larger values waste registers/memory; keep as small as possible

```
next_pow2(x):
  1→1, 2→2, 3→4, 5→8, 9→16, 17→32, 33→64, 65→128, ...
```

### dim1 and dim2_default (Runtime Parameters)

Set in `config()` to the actual problem size:
- `dim1 ≤ D1`: actual number of rows used
- `dim2_default ≤ D2`: actual number of columns per row
- For Partition mode: `dim2_default = 0` (framework handles distribution)

### Choosing D2 for Partition Mode

Since rows have variable length, D2 must accommodate the longest possible row:
```
D2 = max(next_pow2(total_elements / D1 * 2), 64)
```
The `*2` factor provides headroom for unbalanced distributions.

## Shared Memory Sizing

### When to Use Shared Memory

Shared memory provides ~10x faster access than global memory. Use it when:
- Problem has a data matrix (distance, cost, weight)
- The matrix is accessed repeatedly during objective/penalty evaluation

### How to Size

Report the **actual** data size. The framework handles the rest:

```cuda
size_t shared_mem_bytes() const {
    // Distance matrix + demand array
    return (size_t)stride * stride * sizeof(float) + (size_t)n * sizeof(float);
}
```

The framework automatically:
1. If ≤ 48KB: uses default shared memory
2. If 48KB–max_smem: calls `cudaFuncSetAttribute` to extend (GPU-dependent max: T4=64KB, V100=96KB, A100/A800=164KB, H100=228KB)
3. If > max_smem: falls back to global memory, uses `working_set_bytes()` for L2 cache population sizing

### working_set_bytes

Always return the actual data size, regardless of whether it fits in shared memory:

```cuda
size_t working_set_bytes() const {
    return (size_t)n * n * sizeof(float);
}
```

This is used by the framework to auto-calculate population size based on L2 cache capacity.

## RowMode Details

### Single (default)
- `dim1 = 1`, single row of elements
- No cross-row operators
- Simplest and most common

### Fixed
- `dim1 > 1`, all rows have the same length (`dim2_default`)
- Cross-row operators: ROW_SWAP, ROW_REVERSE
- No SPLIT/MERGE (rows cannot change length)
- Use for: JSP (machines × jobs), multi-worker scheduling

### Partition
- `dim1 > 1`, rows have variable length
- Elements are distributed across rows (total count = `total_elements`)
- Cross-row operators: CROSS_RELOCATE, CROSS_SWAP, SEG_RELOCATE, SEG_SWAP, CROSS_EXCHANGE, SPLIT, MERGE
- `cross_row_prob` controls the probability of selecting cross-row operators
- Use for: VRP (vehicles × customers), any partitioning problem

## Quick Reference: Problem → Config

| Problem | Encoding | RowMode | D1 | D2 | cross_row_prob |
|---------|----------|---------|----|----|---------------|
| TSP-50 | Perm | Single | 1 | 64 | 0 |
| TSP-500 | Perm | Single | 1 | 512 | 0 |
| QAP-15 | Perm | Single | 1 | 16 | 0 |
| Assignment-12 | Perm | Single | 1 | 16 | 0 |
| VRP-30-4v | Perm | Partition | 4 | 32 | 0.3 |
| VRPTW-100-25v | Perm | Partition | 32 | 32 | 0.3 |
| Knapsack-100 | Binary | Single | 1 | 128 | 0 |
| Scheduling-20 | Binary | Single | 1 | 32 | 0 |
| Graph Color-50 | Integer | Single | 1 | 64 | 0 |
| JSP-6m-6j | Perm | Fixed | 8 | 8 | 0.2 |
