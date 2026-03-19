---
name: cugenopt-problem-gen
description: Generate cuGenOpt GPU-accelerated optimization Problem definitions from natural language descriptions. Use when the user describes a combinatorial optimization problem (TSP, VRP, knapsack, scheduling, assignment, graph coloring, bin packing, etc.) and wants to solve it with cuGenOpt, or asks to generate, create, model, or run an optimization problem on GPU.
---

# cuGenOpt Problem Generator

Generate cuGenOpt Problem definitions from natural language descriptions. Users describe their optimization problem in plain language; this Skill produces compilable `.cuh` + `main.cu` files and optionally compiles/runs them.

## When to Use

Activate this Skill when the user:
- Describes a combinatorial optimization problem and wants to solve it with cuGenOpt
- Asks to "generate", "create", "write", or "define" a cuGenOpt Problem
- Asks to "solve", "run", or "try" an optimization problem on GPU
- Mentions problem types like TSP, VRP, knapsack, scheduling, assignment, bin packing, graph coloring, etc.
- Asks "how to express / model [problem] in cuGenOpt"

## Prerequisites

The cuGenOpt framework source must be accessible. Locate it by searching for `prototype/core/solver.cuh`:
```
FRAMEWORK_ROOT = directory containing prototype/core/solver.cuh
```
If not found, ask the user for the framework location.

## Workflow

### Step 0: Determine Execution Depth

Infer from the user's wording how far to go:

| Signal | Intent | Depth |
|--------|--------|-------|
| "generate" / "write" / "define" | Code only | Phase 1 → 2 |
| "solve" / "run" / "try" / "see results" | Full run | Phase 1 → 2 → 3 |
| Provides data (CSV, array, file path) | Full run | Phase 1 → 2 → 3 |
| "model" / "express" / "how to" | Code + explanation | Phase 1 → 2 |
| Ambiguous | Code, then ask | Phase 1 → 2, ask |

**Principle**: when in doubt, ask rather than silently consuming GPU.

### Step 1: Analyze the Problem (Phase 1)

Extract from the user's description:

1. **Decision variables** — what is being decided?
2. **Encoding type** — use the decision tree below
3. **Dimensions** — D1, D2, dim1, dim2_default, total_elements
4. **Objective(s)** — minimize/maximize what?
5. **Constraints** — what makes a solution infeasible?
6. **Data** — what input data is needed?

#### Encoding Decision Tree

```
What are the decision variables?
├── Ordering / assignment (no repeats) → Permutation
│   ├── Single sequence (TSP, QAP, assignment) → RowMode::Single, D1=1
│   ├── Multiple equal-length sequences (JSP) → RowMode::Fixed, D1=num_machines
│   └── Variable-length partitions (VRP, VRPTW) → RowMode::Partition, D1=max_vehicles
├── Select / not-select (0-1) → Binary
│   └── Knapsack, scheduling, subset selection → RowMode::Single, D1=1
└── Bounded integer (multi-level) → Integer
    ├── Single row (graph coloring, load balance) → RowMode::Single, D1=1
    └── Multiple equal rows (multi-machine) → RowMode::Fixed, D1=num_machines
```

#### Dimension Calculation

| Parameter | Rule |
|-----------|------|
| `D1` | Template max rows. Single=1; VRP=max_vehicles; JSP=num_machines. Round up to power of 2 if > 1. |
| `D2` | Template max columns. Single=num_elements; Partition=max(num_elements/D1*2, 64). Round up to power of 2. |
| `dim1` | Actual rows at runtime. ≤ D1. |
| `dim2_default` | Single/Fixed: num_elements; Partition: 0 (framework allocates). |
| `total_elements` | Partition only: total number of elements to distribute across rows. |

#### Confirmation Strategy

| Complexity | Criteria | Action |
|------------|----------|--------|
| **Low** | Standard problem with direct reference (TSP, knapsack, assignment) | Proceed without pause |
| **Medium** | Custom constraints (prioritized VRP, skill-matched scheduling) | Generate code, then summarize logic in natural language for user confirmation |
| **High** | Multi-objective, ambiguous description, non-standard encoding | Output problem spec first for confirmation, then generate code |

For medium/high complexity, summarize like:
> "Objective: minimize total distance. Constraints: each vehicle ≤ 100 capacity, penalty = 100 × excess. Encoding: Permutation with Partition (5 vehicles, 30 customers)."

The user confirms the **logic summary**, not the code.

### Step 2: Generate Code (Phase 2)

Generate two files in the user's chosen directory (default: a new folder under the workspace):

#### problem.cuh

```cuda
#pragma once
#include "core/types.cuh"
#include "core/cuda_utils.cuh"
#include "core/operators.cuh"

struct MyProblem : ProblemBase<MyProblem, D1, D2> {
    // GPU data pointers
    const float* d_data;
    int n;

    // === Objective definition ===
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };

    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return /* objective calculation */;
            default: return 0.0f;
        }
    }

    __device__ float compute_penalty(const Sol& sol) const {
        return /* 0.0f if no constraints, else penalty value */;
    }

    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::/* Permutation | Binary | Integer */;
        cfg.dim1 = /* actual rows */;
        cfg.dim2_default = /* actual columns */;
        fill_obj_config(cfg);
        // For multi-row:
        // cfg.row_mode = RowMode::/* Fixed | Partition */;
        // cfg.cross_row_prob = 0.3f;
        // cfg.total_elements = n;  // Partition only
        return cfg;
    }

    // === Shared memory (when data fits) ===
    size_t shared_mem_bytes() const {
        return /* total bytes of data to cache, or 0 if too large */;
    }

    size_t working_set_bytes() const {
        return /* actual data size in bytes (for L2 cache estimation) */;
    }

    __device__ void load_shared(char* smem, int tid, int bsz) {
        // Copy d_data into shared memory, then redirect pointer
    }

    // === Factory ===
    static MyProblem create(/* host data params */) {
        MyProblem prob;
        // cudaMalloc + cudaMemcpy for each data array
        return prob;
    }

    void destroy() {
        // cudaFree each GPU pointer
    }
};
```

#### main.cu

```cuda
#include "core/solver.cuh"
#include "problem.cuh"
#include <cstdio>

int main() {
    // 1. Prepare data (hardcoded or read from file)

    // 2. Create problem
    auto prob = MyProblem::create(/* ... */);

    // 3. Configure solver
    SolverConfig scfg;
    scfg.time_limit_sec = 30.0f;
    scfg.use_aos = true;
    scfg.verbose = true;

    // 4. Solve
    auto result = solve(prob, scfg);

    // 5. Print results
    printf("Best objective: %.4f\n", result.best_solution.objectives[0]);
    printf("Penalty: %.4f\n", result.best_solution.penalty);
    printf("Generations: %d, Time: %.1f ms\n", result.generations, result.elapsed_ms);

    // 6. Print solution details
    // ...

    prob.destroy();
    return 0;
}
```

**Code generation rules**:
- Read `reference/problem-api.md` for the full ProblemBase interface specification
- Read `reference/encoding-guide.md` for encoding selection and dimension calculation details
- Read `reference/examples.md` for end-to-end examples of natural language → code
- `shared_mem_bytes()` should return the actual data size; the framework handles overflow automatically via `cudaFuncSetAttribute`
- `working_set_bytes()` should always return the actual data size (used for population sizing)
- For Partition encoding, set `cfg.dim2_default = 0` and `cfg.total_elements = n`
- Use `CUDA_CHECK()` macro for all CUDA API calls
- Penalty should be proportional to constraint violation magnitude

### Step 3: Validate & Run (Phase 3)

Only enter this phase if execution depth requires it.

#### 3a. Environment Detection

```
1. Run: nvidia-smi
   ├── Success → local/direct mode, go to 3b
   └── Failure → no local GPU
       └── Ask: "No GPU detected locally. Do you have a remote GPU server?"
           ├── User provides SSH info → remote mode
           │   - scp files to remote
           │   - Run all commands via ssh <host> "..."
           ├── User says "I use Cursor Remote SSH" → suggest switching to remote window
           └── No GPU → deliver code only, print compile command

2. Detect compute capability:
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader → sm_XX
```

#### 3b. Compile

```bash
nvcc -O2 -std=c++17 --extended-lambda \
     -arch=sm_XX \
     -I FRAMEWORK_ROOT/prototype \
     -I FRAMEWORK_ROOT/prototype/core \
     main.cu -o solve
```

The two `-I` paths are both needed: `prototype/` for `#include "core/solver.cuh"` in user code, and `prototype/core/` for `#include "types.cuh"` inside framework headers.

If `nvcc` not found:
- Check `ls /usr/local/cuda*/bin/nvcc` — if found, fix PATH
- Otherwise ask: "nvcc not found. Want me to help configure CUDA?"
  - Check and guide: CUDA Toolkit install, g++ install, driver compatibility

If compilation fails: read error, fix code, retry (up to 3 attempts).

#### 3c. Run & Verify

```bash
./solve
```

Verification checklist:
| Check | Method | Pass |
|-------|--------|------|
| No crash | exit code 0 | ✓ |
| penalty = 0 | check output | Feasible solution |
| Objective reasonable | compare to known/estimated optimum | Within expected range |

If penalty > 0: report constraint violations, suggest adjusting penalty weights or constraint logic.

#### 3d. Remote Mode

When using a remote GPU server:
1. Check if framework exists on remote: `ssh <host> "ls <path>/prototype/core/solver.cuh"`
   - Not found → `scp -r FRAMEWORK_ROOT/prototype <host>:<remote_path>/`
2. Upload generated files: `scp problem.cuh main.cu <host>:<remote_path>/`
3. Compile and run via SSH: `ssh <host> "cd <remote_path> && nvcc ... && ./solve"`
4. Capture stdout for results

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `nvcc: command not found` | CUDA not installed or not in PATH | Trigger environment setup |
| `no supported gcc/g++ host compiler` | g++ version incompatible | Install compatible version |
| D2 too small | Elements exceed template parameter | Increase D2 (power of 2) |
| `penalty always > 0` | Constraints too tight or penalty logic wrong | Review compute_penalty |
| Objective is 0 or NaN | Data not loaded to GPU | Check cudaMalloc/cudaMemcpy in create() |
| Very low gens/s | Data matrix too large for shared memory | Check shared_mem_bytes / working_set_bytes |

## Reference Files

For detailed specifications, read these files in the `reference/` directory alongside this SKILL.md:

- **`problem-api.md`** — Complete ProblemBase interface: every method, its signature, when to implement it, and gotchas
- **`encoding-guide.md`** — Encoding type selection, dimension calculation rules, RowMode details, shared memory sizing
- **`examples.md`** — 4 end-to-end examples: natural language input → analysis → generated code
