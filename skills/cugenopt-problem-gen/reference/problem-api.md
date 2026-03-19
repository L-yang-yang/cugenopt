# ProblemBase API Reference

Complete interface specification for `ProblemBase<Derived, D1, D2>` (defined in `core/types.cuh`).

## Template Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `Derived` | struct | The concrete problem type (CRTP pattern) |
| `D1` | int | Maximum number of rows (compile-time constant, power of 2 recommended) |
| `D2` | int | Maximum columns per row (compile-time constant, power of 2 recommended) |

The base class provides:
- `using Sol = Solution<D1, D2>;` — the solution type
- `static constexpr int NUM_OBJ` — auto-derived from `Derived::OBJ_DEFS`
- `evaluate(Sol&)` — calls `compute_obj` for each objective + `compute_penalty`
- `fill_obj_config(ProblemConfig&)` — populates objective fields from `OBJ_DEFS`
- `obj_config()` — returns `ObjConfig` for the solver

## Required Interface

### 1. `OBJ_DEFS` — Objective Definitions (static constexpr)

```cuda
static constexpr ObjDef OBJ_DEFS[] = {
    {ObjDir::Minimize, 1.0f, 0.0f},   // index 0
    // {ObjDir::Maximize, 0.5f, 0.0f}, // index 1 (multi-objective)
};
```

Each `ObjDef`:
- `dir`: `ObjDir::Minimize` or `ObjDir::Maximize`
- `weight`: importance weight for `CompareMode::Weighted` (default mode)
- `tolerance`: tolerance for `CompareMode::Lexicographic`

Most problems have a single objective. Multi-objective (up to 4) is supported.

### 2. `compute_obj` — Objective Calculation

```cuda
__device__ float compute_obj(int idx, const Sol& sol) const;
```

- Runs on GPU (`__device__`)
- `idx` corresponds to `OBJ_DEFS[idx]`
- Use a `switch` statement dispatching to helper functions
- Access solution data via `sol.data[row][col]` and `sol.dim2_sizes[row]`

**Pattern**:
```cuda
__device__ float compute_obj(int idx, const Sol& sol) const {
    switch (idx) {
        case 0: return calc_total_cost(sol);
        default: return 0.0f;
    }
}
```

### 3. `compute_penalty` — Constraint Violation

```cuda
__device__ float compute_penalty(const Sol& sol) const;
```

- Returns `0.0f` for feasible solutions
- Returns a positive value proportional to violation magnitude for infeasible solutions
- The solver always prefers feasible solutions (penalty=0) over infeasible ones
- For multiple constraints, sum all violations

**Guidelines**:
- Scale penalty to be comparable to objective magnitude
- Example: capacity overflow → `(excess_load) * 100.0f`
- Example: vehicle count exceeded → `(excess_vehicles) * 1000.0f`

### 4. `config` — Problem Configuration

```cuda
ProblemConfig config() const;
```

Returns runtime metadata. Must set:

```cuda
ProblemConfig config() const {
    ProblemConfig cfg;
    cfg.encoding = EncodingType::Permutation;  // or Binary, Integer
    cfg.dim1 = /* actual rows used */;
    cfg.dim2_default = /* actual columns */;
    fill_obj_config(cfg);  // auto-fills objectives from OBJ_DEFS

    // Multi-row problems:
    // cfg.row_mode = RowMode::Fixed;       // equal-length rows
    // cfg.row_mode = RowMode::Partition;   // variable-length rows
    // cfg.cross_row_prob = 0.3f;           // cross-row operator probability
    // cfg.total_elements = n;              // Partition: total elements across all rows

    // Integer encoding:
    // cfg.value_lower_bound = 0;
    // cfg.value_upper_bound = num_colors - 1;

    return cfg;
}
```

### 5. `create` / `destroy` — Factory Methods

```cuda
static MyProblem create(/* host-side data */) {
    MyProblem prob;
    prob.n = n;
    // Allocate GPU memory and copy data
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(float) * n * n));
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, sizeof(float) * n * n, cudaMemcpyHostToDevice));
    prob.d_data = d_ptr;
    return prob;
}

void destroy() {
    if (d_data) { cudaFree(const_cast<float*>(d_data)); d_data = nullptr; }
}
```

**Rules**:
- All GPU memory allocated in `create()`, freed in `destroy()`
- Use `CUDA_CHECK()` for every CUDA API call
- Store both `d_` (device) and optionally `h_` (host) pointers
- `const_cast` needed in `destroy()` because pointers are `const float*`

## Optional Interface

### 6. `shared_mem_bytes` — Shared Memory Requirement

```cuda
size_t shared_mem_bytes() const;
```

- Returns total bytes of problem data to cache in shared memory
- Return the **actual** data size; the framework handles overflow:
  - ≤ 48KB: fits default shared memory
  - 48KB–164KB: framework calls `cudaFuncSetAttribute` to extend (GPU-dependent)
  - Too large: framework falls back to global memory automatically
- Default (from base class): returns 0

**Example** (distance matrix):
```cuda
size_t shared_mem_bytes() const {
    return (size_t)n * n * sizeof(float);  // report actual need
}
```

### 7. `working_set_bytes` — Global Memory Working Set

```cuda
size_t working_set_bytes() const;
```

- Returns the per-block hot data size in global memory
- Used by the framework to estimate L2 cache pressure and auto-size population
- Default: returns `shared_mem_bytes()`
- **Override when** `shared_mem_bytes()` returns 0 (data doesn't fit in shared memory) — return the actual data size so population sizing works correctly

**Example**:
```cuda
size_t working_set_bytes() const {
    return (size_t)n * n * sizeof(float) + (size_t)n * sizeof(float);
}
```

### 8. `load_shared` — Load Data into Shared Memory

```cuda
__device__ void load_shared(char* smem, int tid, int bsz);
```

- Called by framework when `shared_mem_bytes() > 0`
- Copy data from global memory to shared memory using cooperative loading
- **Redirect the device pointer** to shared memory after loading

**Pattern**:
```cuda
__device__ void load_shared(char* smem, int tid, int bsz) {
    float* s_data = reinterpret_cast<float*>(smem);
    int total = n * n;
    for (int i = tid; i < total; i += bsz)
        s_data[i] = d_data[i];
    d_data = s_data;  // redirect pointer to shared memory
}
```

For multiple arrays, lay them out sequentially in `smem`:
```cuda
__device__ void load_shared(char* smem, int tid, int bsz) {
    float* s_dist = reinterpret_cast<float*>(smem);
    int dist_size = stride * stride;
    for (int i = tid; i < dist_size; i += bsz) s_dist[i] = d_dist[i];
    d_dist = s_dist;

    float* s_demand = s_dist + dist_size;
    for (int i = tid; i < n; i += bsz) s_demand[i] = d_demand[i];
    d_demand = s_demand;
}
```

### 9. `heuristic_matrices` — Data for Heuristic Initialization

```cuda
int heuristic_matrices(HeuristicMatrix* out, int max_count) const;
```

- Returns host-side matrices for constructing heuristic initial solutions
- The framework sorts elements by row/column sums to generate better-than-random starting points
- Return value: number of matrices provided (0 = no heuristic init)

**Example** (distance matrix for TSP):
```cuda
int heuristic_matrices(HeuristicMatrix* out, int max_count) const {
    if (max_count < 1 || !h_dist) return 0;
    out[0] = {h_dist, n};
    return 1;
}
```

### 10. `init_relation_matrix` — G/O Matrix for Guided Rebuild

```cuda
void init_relation_matrix(float* h_G, float* h_O, int N) const;
```

- Provides prior knowledge for the LNS guided rebuild operator
- `G[i*N+j]`: grouping tendency (symmetric, higher = more likely in same group)
- `O[i*N+j]`: ordering tendency (asymmetric, higher = i before j)
- Values in [0, 1], typically scaled from problem data (e.g., distance proximity)
- Default: does nothing (matrices stay zero, learned from search history)

## Solution Data Access

```cuda
sol.data[row][col]      // element value at (row, col)
sol.dim2_sizes[row]     // actual length of row (may be < D2)
sol.objectives[idx]     // objective value (set by evaluate())
sol.penalty             // penalty value (set by evaluate())
```

- **Permutation (Single)**: `sol.data[0][0..n-1]` contains a permutation of `0..n-1`
- **Permutation (Partition)**: `sol.data[r][0..sol.dim2_sizes[r]-1]` for each route/partition
- **Binary**: `sol.data[0][i]` is 0 or 1
- **Integer**: `sol.data[0][i]` is in `[value_lower_bound, value_upper_bound]`

## Key Types Reference

```cuda
enum class EncodingType { Permutation, Binary, Integer };
enum class RowMode { Single, Fixed, Partition };
enum class ObjDir { Minimize, Maximize };
enum class CompareMode { Weighted, Lexicographic };

struct ObjDef { ObjDir dir; float weight; float tolerance; };
struct HeuristicMatrix { const float* data; int N; };

struct ProblemConfig {
    EncodingType encoding;
    int dim1, dim2_default, num_objectives;
    ObjDir obj_dirs[4]; float obj_weights[4];
    CompareMode compare_mode;
    RowMode row_mode;
    float cross_row_prob;
    int total_elements;
    int value_lower_bound, value_upper_bound;
};

struct SolverConfig {
    int pop_size;           // 0 = auto
    int max_gen;            // max generations
    float time_limit_sec;   // 0 = no limit
    bool use_aos;           // adaptive operator selection
    bool verbose;
    unsigned seed;
};
```
