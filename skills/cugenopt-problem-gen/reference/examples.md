# End-to-End Examples

Four complete examples from natural language description to generated code.

---

## Example 1: 0-1 Knapsack (Low Complexity)

### User Input
> "I have 8 items with weights [2,3,4,5,9,7,8,6] and values [3,4,5,8,10,7,9,6]. Knapsack capacity is 20. Maximize total value."

### Analysis
- **Decision**: select or not → **Binary**
- **RowMode**: Single (D1=1)
- **D2**: next_pow2(8) = 8
- **Objective**: Maximize total value
- **Constraint**: total weight ≤ 20
- **Complexity**: Low (standard knapsack, direct reference)

### Generated: problem.cuh

```cuda
#pragma once
#include "core/types.cuh"
#include "core/cuda_utils.cuh"
#include "core/operators.cuh"

struct Knapsack8 : ProblemBase<Knapsack8, 1, 8> {
    const float* d_weights;
    const float* d_values;
    float capacity;
    int n;

    __device__ float calc_total_value(const Sol& sol) const {
        float tv = 0.0f;
        const int* sel = sol.data[0];
        for (int i = 0; i < n; i++)
            if (sel[i]) tv += d_values[i];
        return tv;
    }

    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Maximize, 1.0f, 0.0f},
    };

    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_total_value(sol);
            default: return 0.0f;
        }
    }

    __device__ float compute_penalty(const Sol& sol) const {
        float tw = 0.0f;
        const int* sel = sol.data[0];
        for (int i = 0; i < n; i++)
            if (sel[i]) tw += d_weights[i];
        float over = tw - capacity;
        return (over > 0.0f) ? over * 50.0f : 0.0f;
    }

    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Binary;
        cfg.dim1 = 1;
        cfg.dim2_default = n;
        fill_obj_config(cfg);
        return cfg;
    }

    size_t shared_mem_bytes() const {
        return 2 * (size_t)n * sizeof(float);
    }

    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sw = reinterpret_cast<float*>(smem);
        float* sv = sw + n;
        for (int i = tid; i < n; i += bsz) {
            sw[i] = d_weights[i];
            sv[i] = d_values[i];
        }
        d_weights = sw;
        d_values = sv;
    }

    static Knapsack8 create(const float* hw, const float* hv, int n, float cap) {
        Knapsack8 prob;
        prob.n = n;
        prob.capacity = cap;
        float *dw, *dv;
        CUDA_CHECK(cudaMalloc(&dw, sizeof(float) * n));
        CUDA_CHECK(cudaMalloc(&dv, sizeof(float) * n));
        CUDA_CHECK(cudaMemcpy(dw, hw, sizeof(float) * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv, hv, sizeof(float) * n, cudaMemcpyHostToDevice));
        prob.d_weights = dw;
        prob.d_values = dv;
        return prob;
    }

    void destroy() {
        if (d_weights) cudaFree(const_cast<float*>(d_weights));
        if (d_values) cudaFree(const_cast<float*>(d_values));
        d_weights = nullptr;
        d_values = nullptr;
    }
};
```

### Generated: main.cu

```cuda
#include "core/solver.cuh"
#include "problem.cuh"
#include <cstdio>

int main() {
    const int n = 8;
    float weights[] = {2, 3, 4, 5, 9, 7, 8, 6};
    float values[]  = {3, 4, 5, 8, 10, 7, 9, 6};
    float capacity = 20.0f;

    auto prob = Knapsack8::create(weights, values, n, capacity);

    SolverConfig scfg;
    scfg.time_limit_sec = 5.0f;
    scfg.use_aos = true;
    scfg.verbose = true;

    auto result = solve(prob, scfg);

    printf("Best value: %.2f\n", result.best_solution.objectives[0]);
    printf("Penalty: %.2f\n", result.best_solution.penalty);
    printf("Selected items: ");
    for (int i = 0; i < n; i++)
        if (result.best_solution.data[0][i]) printf("%d ", i);
    printf("\n");

    prob.destroy();
    return 0;
}
```

---

## Example 2: Assignment Problem (Low Complexity)

### User Input
> "Assign 10 workers to 10 tasks. Cost matrix is in a file `cost_10x10.txt`. Minimize total cost."

### Analysis
- **Decision**: assign each worker to a unique task → **Permutation**
- **RowMode**: Single (D1=1)
- **D2**: next_pow2(10) = 16
- **Objective**: Minimize total cost
- **Constraint**: none (permutation encoding guarantees one-to-one)
- **Data**: read from file
- **Complexity**: Low (standard assignment)

### Generated: problem.cuh

```cuda
#pragma once
#include "core/types.cuh"
#include "core/cuda_utils.cuh"
#include "core/operators.cuh"

struct Assignment10 : ProblemBase<Assignment10, 1, 16> {
    const float* d_cost;
    int n;

    __device__ float calc_total_cost(const Sol& sol) const {
        float total = 0.0f;
        const int* assign = sol.data[0];
        for (int i = 0; i < n; i++)
            total += d_cost[i * n + assign[i]];
        return total;
    }

    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };

    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_total_cost(sol);
            default: return 0.0f;
        }
    }

    __device__ float compute_penalty(const Sol& sol) const {
        return 0.0f;
    }

    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = 1;
        cfg.dim2_default = n;
        fill_obj_config(cfg);
        return cfg;
    }

    size_t shared_mem_bytes() const {
        return (size_t)n * n * sizeof(float);
    }

    size_t working_set_bytes() const {
        return (size_t)n * n * sizeof(float);
    }

    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sc = reinterpret_cast<float*>(smem);
        int total = n * n;
        for (int i = tid; i < total; i += bsz) sc[i] = d_cost[i];
        d_cost = sc;
    }

    static Assignment10 create(const float* hc, int n) {
        Assignment10 prob;
        prob.n = n;
        float* dc;
        CUDA_CHECK(cudaMalloc(&dc, sizeof(float) * n * n));
        CUDA_CHECK(cudaMemcpy(dc, hc, sizeof(float) * n * n, cudaMemcpyHostToDevice));
        prob.d_cost = dc;
        return prob;
    }

    void destroy() {
        if (d_cost) { cudaFree(const_cast<float*>(d_cost)); d_cost = nullptr; }
    }
};
```

### Generated: main.cu

```cuda
#include "core/solver.cuh"
#include "problem.cuh"
#include <cstdio>
#include <cstdlib>

int main() {
    const int n = 10;
    float cost[n * n];

    FILE* f = fopen("cost_10x10.txt", "r");
    if (!f) { fprintf(stderr, "Cannot open cost_10x10.txt\n"); return 1; }
    for (int i = 0; i < n * n; i++) fscanf(f, "%f", &cost[i]);
    fclose(f);

    auto prob = Assignment10::create(cost, n);

    SolverConfig scfg;
    scfg.time_limit_sec = 10.0f;
    scfg.use_aos = true;
    scfg.verbose = true;

    auto result = solve(prob, scfg);

    printf("Best cost: %.2f\n", result.best_solution.objectives[0]);
    printf("Assignment: ");
    for (int i = 0; i < n; i++)
        printf("worker %d → task %d  ", i, result.best_solution.data[0][i]);
    printf("\n");

    prob.destroy();
    return 0;
}
```

---

## Example 3: Vehicle Routing with Capacity (Medium Complexity)

### User Input
> "I have 1 depot and 30 customers. 4 trucks, each with capacity 100. Customer coordinates and demands are in `customers.csv` (columns: id, x, y, demand). Minimize total travel distance."

### Analysis
- **Decision**: assign customers to trucks and determine visit order → **Permutation**
- **RowMode**: Partition (variable-length routes)
- **D1**: next_pow2(4) = 4
- **D2**: max(next_pow2(30/4*2), 64) = 64
- **Objective**: Minimize total distance (depot → customers → depot for each truck)
- **Constraint**: each truck's total demand ≤ 100
- **Data**: CSV with coordinates → compute distance matrix
- **Complexity**: Medium (custom constraint, Partition encoding)

### Logic Summary (for user confirmation)
> "Objective: minimize total travel distance across all trucks. Each truck starts and ends at depot (id=0). Constraint: total demand per truck ≤ 100, penalty = 100 × excess. Encoding: Permutation with Partition, 4 trucks, 30 customers."

### Generated: problem.cuh

```cuda
#pragma once
#include "core/types.cuh"
#include "core/cuda_utils.cuh"
#include "core/operators.cuh"
#include <cmath>

struct VRP30 : ProblemBase<VRP30, 4, 64> {
    const float* d_dist;    // (n+1)×(n+1) distance matrix including depot
    const float* d_demand;  // n customer demands
    int n;                  // number of customers (excluding depot)
    int stride;             // n+1
    float capacity;
    int num_vehicles;

    __device__ float compute_route_dist(const int* route, int size) const {
        if (size == 0) return 0.0f;
        float dist = 0.0f;
        int prev = 0;  // depot
        for (int j = 0; j < size; j++) {
            int node = route[j] + 1;  // customer indices are 0-based, node indices 1-based
            dist += d_dist[prev * stride + node];
            prev = node;
        }
        dist += d_dist[prev * stride + 0];  // return to depot
        return dist;
    }

    __device__ float calc_total_distance(const Sol& sol) const {
        float total = 0.0f;
        for (int r = 0; r < num_vehicles; r++)
            total += compute_route_dist(sol.data[r], sol.dim2_sizes[r]);
        return total;
    }

    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };

    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_total_distance(sol);
            default: return 0.0f;
        }
    }

    __device__ float compute_penalty(const Sol& sol) const {
        float penalty = 0.0f;
        for (int r = 0; r < num_vehicles; r++) {
            float load = 0.0f;
            for (int j = 0; j < sol.dim2_sizes[r]; j++)
                load += d_demand[sol.data[r][j]];
            if (load > capacity)
                penalty += (load - capacity) * 100.0f;
        }
        return penalty;
    }

    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = num_vehicles;
        cfg.dim2_default = 0;
        fill_obj_config(cfg);
        cfg.row_mode = RowMode::Partition;
        cfg.cross_row_prob = 0.3f;
        cfg.total_elements = n;
        return cfg;
    }

    size_t shared_mem_bytes() const {
        return (size_t)stride * stride * sizeof(float) + (size_t)n * sizeof(float);
    }

    size_t working_set_bytes() const {
        return (size_t)stride * stride * sizeof(float) + (size_t)n * sizeof(float);
    }

    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sd = reinterpret_cast<float*>(smem);
        int dist_size = stride * stride;
        for (int i = tid; i < dist_size; i += bsz) sd[i] = d_dist[i];
        d_dist = sd;
        float* sdem = sd + dist_size;
        for (int i = tid; i < n; i += bsz) sdem[i] = d_demand[i];
        d_demand = sdem;
    }

    static VRP30 create(const float* h_dist, const float* h_demand,
                         int n, float capacity, int num_vehicles) {
        VRP30 prob;
        prob.n = n;
        prob.stride = n + 1;
        prob.capacity = capacity;
        prob.num_vehicles = num_vehicles;

        int nodes = n + 1;
        float* dd;
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * nodes * nodes));
        CUDA_CHECK(cudaMemcpy(dd, h_dist, sizeof(float) * nodes * nodes, cudaMemcpyHostToDevice));
        prob.d_dist = dd;

        float* ddem;
        CUDA_CHECK(cudaMalloc(&ddem, sizeof(float) * n));
        CUDA_CHECK(cudaMemcpy(ddem, h_demand, sizeof(float) * n, cudaMemcpyHostToDevice));
        prob.d_demand = ddem;

        return prob;
    }

    void destroy() {
        if (d_dist) { cudaFree(const_cast<float*>(d_dist)); d_dist = nullptr; }
        if (d_demand) { cudaFree(const_cast<float*>(d_demand)); d_demand = nullptr; }
    }
};
```

### Generated: main.cu

```cuda
#include "core/solver.cuh"
#include "problem.cuh"
#include <cstdio>
#include <cmath>

int main() {
    const int n = 30;
    const int num_vehicles = 4;
    const float capacity = 100.0f;

    float x[n + 1], y[n + 1], demand[n];

    FILE* f = fopen("customers.csv", "r");
    if (!f) { fprintf(stderr, "Cannot open customers.csv\n"); return 1; }

    char header[256];
    fgets(header, sizeof(header), f);  // skip header

    // Read depot (id=0)
    int id;
    fscanf(f, "%d,%f,%f,%*f", &id, &x[0], &y[0]);  // depot has no demand

    // Read customers
    for (int i = 0; i < n; i++) {
        fscanf(f, "%d,%f,%f,%f", &id, &x[i + 1], &y[i + 1], &demand[i]);
    }
    fclose(f);

    // Compute distance matrix
    int nodes = n + 1;
    float dist[nodes * nodes];
    for (int i = 0; i < nodes; i++)
        for (int j = 0; j < nodes; j++) {
            float dx = x[i] - x[j], dy = y[i] - y[j];
            dist[i * nodes + j] = sqrtf(dx * dx + dy * dy);
        }

    auto prob = VRP30::create(dist, demand, n, capacity, num_vehicles);

    SolverConfig scfg;
    scfg.time_limit_sec = 30.0f;
    scfg.use_aos = true;
    scfg.verbose = true;

    auto result = solve(prob, scfg);

    printf("Best distance: %.2f\n", result.best_solution.objectives[0]);
    printf("Penalty: %.2f\n", result.best_solution.penalty);
    for (int r = 0; r < num_vehicles; r++) {
        printf("Truck %d: depot", r);
        for (int j = 0; j < result.best_solution.dim2_sizes[r]; j++)
            printf(" → %d", result.best_solution.data[r][j] + 1);
        printf(" → depot\n");
    }

    prob.destroy();
    return 0;
}
```

---

## Example 4: Graph Coloring (Low Complexity)

### User Input
> "Color a graph with 20 nodes using at most 4 colors. Edges: (0,1),(0,2),(1,3),(2,3),(3,4),...  Minimize the number of colors used, with no two adjacent nodes sharing a color."

### Analysis
- **Decision**: assign a color (0–3) to each node → **Integer**
- **RowMode**: Single (D1=1)
- **D2**: next_pow2(20) = 32
- **Objective**: Minimize number of distinct colors used
- **Constraint**: adjacent nodes must have different colors
- **Complexity**: Low (standard graph coloring)

### Generated: problem.cuh

```cuda
#pragma once
#include "core/types.cuh"
#include "core/cuda_utils.cuh"
#include "core/operators.cuh"

struct GraphColor20 : ProblemBase<GraphColor20, 1, 32> {
    const int* d_adj;     // adjacency matrix n×n (1=edge, 0=no edge)
    int n;
    int max_colors;

    __device__ float calc_num_colors(const Sol& sol) const {
        int used[4] = {0, 0, 0, 0};
        const int* colors = sol.data[0];
        for (int i = 0; i < n; i++) {
            int c = colors[i];
            if (c >= 0 && c < max_colors) used[c] = 1;
        }
        float count = 0.0f;
        for (int c = 0; c < max_colors; c++) count += used[c];
        return count;
    }

    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };

    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_num_colors(sol);
            default: return 0.0f;
        }
    }

    __device__ float compute_penalty(const Sol& sol) const {
        float conflicts = 0.0f;
        const int* colors = sol.data[0];
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (d_adj[i * n + j] && colors[i] == colors[j])
                    conflicts += 1.0f;
        return conflicts * 10.0f;
    }

    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Integer;
        cfg.dim1 = 1;
        cfg.dim2_default = n;
        cfg.value_lower_bound = 0;
        cfg.value_upper_bound = max_colors - 1;
        fill_obj_config(cfg);
        return cfg;
    }

    size_t shared_mem_bytes() const {
        return (size_t)n * n * sizeof(int);
    }

    size_t working_set_bytes() const {
        return (size_t)n * n * sizeof(int);
    }

    __device__ void load_shared(char* smem, int tid, int bsz) {
        int* sa = reinterpret_cast<int*>(smem);
        int total = n * n;
        for (int i = tid; i < total; i += bsz) sa[i] = d_adj[i];
        d_adj = sa;
    }

    static GraphColor20 create(const int* h_adj, int n, int max_colors) {
        GraphColor20 prob;
        prob.n = n;
        prob.max_colors = max_colors;
        int* da;
        CUDA_CHECK(cudaMalloc(&da, sizeof(int) * n * n));
        CUDA_CHECK(cudaMemcpy(da, h_adj, sizeof(int) * n * n, cudaMemcpyHostToDevice));
        prob.d_adj = da;
        return prob;
    }

    void destroy() {
        if (d_adj) { cudaFree(const_cast<int*>(d_adj)); d_adj = nullptr; }
    }
};
```

### Generated: main.cu

```cuda
#include "core/solver.cuh"
#include "problem.cuh"
#include <cstdio>

int main() {
    const int n = 20;
    const int max_colors = 4;

    int adj[n * n] = {0};
    // Define edges
    int edges[][2] = {{0,1},{0,2},{1,3},{2,3},{3,4},
                       {4,5},{5,6},{6,7},{7,8},{8,9},
                       {9,10},{10,11},{11,12},{12,13},{13,14},
                       {14,15},{15,16},{16,17},{17,18},{18,19},
                       {0,19},{1,4},{2,5},{6,9},{7,10}};
    int num_edges = sizeof(edges) / sizeof(edges[0]);
    for (int e = 0; e < num_edges; e++) {
        int u = edges[e][0], v = edges[e][1];
        adj[u * n + v] = 1;
        adj[v * n + u] = 1;
    }

    auto prob = GraphColor20::create(adj, n, max_colors);

    SolverConfig scfg;
    scfg.time_limit_sec = 10.0f;
    scfg.use_aos = true;
    scfg.verbose = true;

    auto result = solve(prob, scfg);

    printf("Colors used: %.0f\n", result.best_solution.objectives[0]);
    printf("Conflicts (penalty): %.2f\n", result.best_solution.penalty);
    printf("Coloring: ");
    for (int i = 0; i < n; i++)
        printf("node%d=%d ", i, result.best_solution.data[0][i]);
    printf("\n");

    prob.destroy();
    return 0;
}
```
