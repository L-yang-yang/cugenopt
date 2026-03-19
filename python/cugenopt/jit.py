"""
JIT compiler for custom cuGenOpt problems.

Workflow:
  1. User provides CUDA code snippets (compute_obj, compute_penalty) + data arrays
  2. Python fills the .cu template with user code
  3. nvcc compiles to executable (cached by content hash)
  4. subprocess runs executable, parses JSON output
"""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cugenopt.validation import (
    CuGenOptCompileError,
    CuGenOptValidationError,
    validate_cuda_snippet,
    validate_data_dict,
    validate_encoding,
    validate_positive_int,
)

_TEMPLATE_PATH = Path(__file__).parent / "jit_template.cu"
_CACHE_DIR = Path.home() / ".cache" / "cugenopt" / "jit"


def _detect_framework_root() -> Path:
    """Find the cuGenOpt framework root (contains core/types.cuh).

    Search order:
      1. Bundled headers inside the installed package (pip install wheel)
      2. Development layout (source tree)
      3. CUGENOPT_ROOT env var
    """
    pkg_dir = Path(__file__).parent  # cugenopt/

    # 1. Bundled headers (wheel layout: cugenopt/include/core/types.cuh)
    bundled = pkg_dir / "include"
    if (bundled / "core" / "types.cuh").exists():
        return bundled

    # 2. Development layout (python/../prototype)
    dev_root = pkg_dir.parent  # python/
    for rel in ["../prototype", "../../prototype"]:
        c = (dev_root / rel).resolve()
        if (c / "core" / "types.cuh").exists():
            return c

    # 3. CUGENOPT_ROOT env var
    env_root = os.environ.get("CUGENOPT_ROOT")
    if env_root:
        p = Path(env_root) / "prototype"
        if (p / "core" / "types.cuh").exists():
            return p

    raise RuntimeError(
        "Cannot find cuGenOpt framework headers. "
        "Set CUGENOPT_ROOT env var to the generic_solver directory, "
        "or reinstall: pip install cugenopt"
    )


def _detect_cuda_arch() -> str:
    """Detect GPU compute capability via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True
        ).strip().split("\n")[0]
        major, minor = out.strip().split(".")
        return f"sm_{major}{minor}"
    except Exception:
        return "sm_75"


def _is_windows() -> bool:
    return os.name == "nt"


def _nvcc_name() -> str:
    return "nvcc.exe" if _is_windows() else "nvcc"


def _find_nvcc() -> str:
    """Find nvcc binary. Search order: PATH → pip-installed → common system paths."""
    # 1. System PATH
    nvcc = shutil.which(_nvcc_name())
    if nvcc:
        return nvcc

    # 2. pip-installed nvidia-cuda-nvcc / nvidia-cuda-nvcc-cu12
    import sys
    import site
    pip_search_dirs = [sys.prefix, *site.getsitepackages(), site.getusersitepackages()]
    nvcc_bin = _nvcc_name()
    for base in pip_search_dirs:
        pip_subpaths = [
            os.path.join("nvidia", "cuda_nvcc", "bin", nvcc_bin),
            os.path.join("nvidia", "cu13", "bin", nvcc_bin),
            os.path.join("nvidia", "cu12", "bin", nvcc_bin),
            os.path.join("bin", nvcc_bin),
        ]
        if _is_windows():
            pip_subpaths += [
                os.path.join("Scripts", nvcc_bin),
                os.path.join("Library", "bin", nvcc_bin),
            ]
        for subpath in pip_subpaths:
            candidate = os.path.join(base, subpath)
            if os.path.isfile(candidate):
                return candidate

    # 3. Common system paths
    if _is_windows():
        cuda_base = os.path.join(os.environ.get("CUDA_PATH", ""),
                                 "bin", "nvcc.exe")
        if os.path.isfile(cuda_base):
            return cuda_base
        for ver in ["12.4", "12.6", "12.0", "11.8"]:
            candidate = os.path.join(
                "C:\\", "Program Files", "NVIDIA GPU Computing Toolkit",
                "CUDA", f"v{ver}", "bin", "nvcc.exe")
            if os.path.isfile(candidate):
                return candidate
    else:
        for candidate in [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-12.4/bin/nvcc",
            "/usr/local/cuda-12.6/bin/nvcc",
            "/usr/local/cuda-12.0/bin/nvcc",
            "/usr/local/cuda-11.8/bin/nvcc",
        ]:
            if os.path.isfile(candidate):
                return candidate

    raise RuntimeError(
        "nvcc not found. Install the CUDA compiler:\n"
        "  pip install nvidia-cuda-nvcc-cu12\n"
        "Or install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    )


def _content_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _fallback_compile_without_bad_ops(
    custom_operators: list,
    encoding: str,
    template: str,
    replacements: dict,
    base_cmd: list,
    fw_root: Path,
    original_stderr: str,
) -> tuple:
    """When compilation fails with custom operators, try excluding them one by one.

    Strategy:
      1. Try compiling without ALL custom operators (baseline)
      2. If baseline compiles, try adding operators back one by one
      3. Report which operators were disabled

    Returns (source, exe_path, cu_path) of the successful compilation.
    Raises CuGenOptCompileError if even baseline fails.
    """
    import warnings
    from cugenopt.operators import generate_custom_operator_cuda

    # Try baseline (no custom ops)
    baseline_replacements = dict(replacements)
    baseline_replacements["{{CUSTOM_OP_DEFINES}}"] = ""
    baseline_replacements["{{CUSTOM_OP_SWITCH}}"] = ""
    baseline_replacements["{{CUSTOM_OP_REGISTRY}}"] = ""

    baseline_source = template
    for key, val in baseline_replacements.items():
        baseline_source = baseline_source.replace(key, val)

    h = _content_hash(baseline_source)
    cu_path = _CACHE_DIR / f"custom_{h}.cu"
    exe_suffix = ".exe" if _is_windows() else ""
    exe_path = _CACHE_DIR / f"custom_{h}{exe_suffix}"

    if not exe_path.exists():
        cu_path.write_text(baseline_source)
        cmd = base_cmd + [str(cu_path), "-o", str(exe_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise CuGenOptCompileError(original_stderr, str(cu_path))

    disabled_names = [op.name for op in custom_operators]
    warnings.warn(
        f"Custom operator(s) caused compilation failure. "
        f"Disabled: {', '.join(disabled_names)}. "
        f"Solving with built-in operators only.\n"
        f"Fix your operator code and retry. "
        f"Original error:\n{original_stderr[:500]}",
        RuntimeWarning,
        stacklevel=4,
    )
    return baseline_source, exe_path, cu_path


def _build_custom_op_defines(switch_block: str, registry_block: str) -> str:
    """Build #define to enable custom operator hooks in framework headers."""
    return "#define CUGENOPT_HAS_CUSTOM_OPS"


def _write_binary_float(path: str, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def _write_binary_int(path: str, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    with open(path, "wb") as f:
        f.write(arr.tobytes())


class DataArray:
    """Describes a data array to be passed to the custom problem."""

    def __init__(self, name: str, data: np.ndarray, dtype: str = "float"):
        self.name = name
        self.data = data
        self.dtype = dtype  # "float" or "int"
        self.size = data.size  # total element count


def _build_encoding_str(encoding: str) -> str:
    mapping = {
        "permutation": "EncodingType::Permutation",
        "binary": "EncodingType::Binary",
        "integer": "EncodingType::Integer",
    }
    return mapping.get(encoding.lower(), "EncodingType::Permutation")


def _build_row_mode_str(row_mode: str) -> str:
    mapping = {
        "single": "RowMode::Single",
        "fixed": "RowMode::Fixed",
        "partition": "RowMode::Partition",
    }
    return mapping.get(row_mode.lower(), "RowMode::Single")


def _build_obj_dir_str(direction: str) -> str:
    return "ObjDir::Maximize" if direction.lower().startswith("max") else "ObjDir::Minimize"


def compile_and_solve(
    compute_obj: str,
    compute_penalty: str = "return 0.0f;",
    data: Optional[Dict[str, np.ndarray]] = None,
    int_data: Optional[Dict[str, np.ndarray]] = None,
    encoding: str = "permutation",
    dim1: int = 1,
    dim2: int = 64,
    n: Optional[int] = None,
    row_mode: str = "single",
    total_elements: int = 0,
    cross_row_prob: float = 0.0,
    perm_repeat_count: int = 1,
    value_lower: int = 0,
    value_upper: int = 1,
    objectives: Optional[List[Tuple[str, float]]] = None,
    shared_mem: Optional[str] = None,
    load_shared: Optional[str] = None,
    pop_size: int = 0,
    max_gen: int = 1000,
    time_limit: float = 0.0,
    seed: int = 42,
    use_aos: bool = False,
    sa_temp_init: float = 0.0,
    verbose: bool = False,
    framework_root: Optional[str] = None,
    cuda_arch: Optional[str] = None,
    custom_operators: Optional[List] = None,
) -> Dict[str, Any]:
    """
    JIT-compile and solve a custom optimization problem.

    Args:
        compute_obj: CUDA code for the compute_obj function body.
            Available variables: idx (objective index), sol (const Sol&),
            and any data fields you declared.
        compute_penalty: CUDA code for compute_penalty body.
        data: Dict of name -> numpy float32 array for problem data.
        int_data: Dict of name -> numpy int32 array for problem data.
        encoding: "permutation", "binary", or "integer".
        dim1: Number of rows in solution (1 for most problems).
        dim2: Max columns per row.
        n: Problem size (number of elements). If None, inferred from data.
        row_mode: "single", "fixed", or "partition".
        total_elements: For partition mode, total elements across all rows.
        cross_row_prob: Probability of cross-row operations.
        perm_repeat_count: For multiset permutation (JSP-style).
        value_lower, value_upper: Bounds for integer encoding.
        objectives: List of (direction, weight) tuples. Default: [("minimize", 1.0)].
        shared_mem: Expression for shared_mem_bytes() return value.
            Use '_n' for problem size. E.g. "(size_t)_n * _n * sizeof(float)"
        load_shared: CUDA code for load_shared body.
        pop_size, max_gen, time_limit, seed, use_aos, sa_temp_init, verbose:
            Solver configuration parameters.
        framework_root: Path to cuGenOpt framework. Auto-detected if None.
        cuda_arch: CUDA architecture (e.g. "sm_75"). Auto-detected if None.

    Returns:
        Dict with keys: objective, penalty, solution, elapsed_ms, generations,
        stop_reason, objectives.

    Example:
        >>> result = compile_and_solve(
        ...     compute_obj='''
        ...         if (idx != 0) return 0.0f;
        ...         float total = 0.0f;
        ...         const int* route = sol.data[0];
        ...         int size = sol.dim2_sizes[0];
        ...         for (int i = 0; i < size; i++)
        ...             total += d_dist[route[i] * _n + route[(i+1) % size]];
        ...         return total;
        ...     ''',
        ...     data={"d_dist": dist_matrix},
        ...     encoding="permutation", dim2=64, n=20,
        ...     time_limit=5.0,
        ... )
    """
    if data is None:
        data = {}
    if int_data is None:
        int_data = {}
    if objectives is None:
        objectives = [("minimize", 1.0)]

    # --- Input validation ---
    compute_obj = validate_cuda_snippet(compute_obj, "compute_obj")
    if compute_penalty != "return 0.0f;":
        compute_penalty = validate_cuda_snippet(compute_penalty, "compute_penalty")
    encoding = validate_encoding(encoding)
    dim1 = validate_positive_int(dim1, "dim1")
    dim2 = validate_positive_int(dim2, "dim2")
    max_gen = validate_positive_int(max_gen, "max_gen")
    if data:
        data = validate_data_dict(data, "float")
    if int_data:
        int_data = validate_data_dict(int_data, "int")

    # Infer n from data if not provided
    if n is None:
        for arr in data.values():
            if arr.ndim == 2:
                n = arr.shape[0]
                break
            elif arr.ndim == 1:
                n = arr.shape[0]
                break
        if n is None:
            n = dim2
    n = validate_positive_int(n, "n")

    # Framework root
    fw_root = Path(framework_root) if framework_root else _detect_framework_root()

    # Read template
    template = _TEMPLATE_PATH.read_text()

    # Build data fields
    all_data = []
    data_fields_lines = []
    for name, arr in data.items():
        all_data.append(DataArray(name, arr, "float"))
        data_fields_lines.append(f"    const float* {name};")
    for name, arr in int_data.items():
        all_data.append(DataArray(name, arr, "int"))
        data_fields_lines.append(f"    const int* {name};")
    data_fields = "\n".join(data_fields_lines) if data_fields_lines else "    // no data fields"

    # Build OBJ_DEFS
    obj_defs_parts = []
    for direction, weight in objectives:
        obj_defs_parts.append(f"{{{_build_obj_dir_str(direction)}, {weight}f, 0.0f}}")
    obj_defs = ", ".join(obj_defs_parts)

    # Build shared memory
    if shared_mem is None:
        total_bytes_parts = []
        for da in all_data:
            elem_size = "sizeof(float)" if da.dtype == "float" else "sizeof(int)"
            total_bytes_parts.append(f"(size_t){da.size} * {elem_size}")
        if total_bytes_parts:
            total_expr = " + ".join(total_bytes_parts)
            shared_mem_expr = f"size_t need = {total_expr};\n        return (need <= 48 * 1024) ? need : 0;"
        else:
            shared_mem_expr = "return 0;"
    else:
        shared_mem_expr = f"size_t need = {shared_mem};\n        return (need <= 48 * 1024) ? need : 0;"

    # Build load_shared
    if load_shared is None:
        load_lines = []
        offset = "smem"
        for i, da in enumerate(all_data):
            ctype = "float" if da.dtype == "float" else "int"
            ptr_name = f"s_{da.name}"
            if i == 0:
                load_lines.append(f"        {ctype}* {ptr_name} = reinterpret_cast<{ctype}*>(smem);")
            else:
                prev = all_data[i - 1]
                prev_ptr = f"s_{prev.name}"
                load_lines.append(f"        {ctype}* {ptr_name} = reinterpret_cast<{ctype}*>({prev_ptr} + {prev.size});")
            load_lines.append(f"        for (int i = tid; i < {da.size}; i += bsz) {ptr_name}[i] = {da.name}[i];")
            load_lines.append(f"        {da.name} = {ptr_name};")
        load_shared_body = "\n".join(load_lines) if load_lines else "        // no data to load"
    else:
        load_shared_body = load_shared

    # Build destroy body
    destroy_lines = []
    for da in all_data:
        ctype = "float" if da.dtype == "float" else "int"
        destroy_lines.append(f"        if ({da.name}) cudaFree(const_cast<{ctype}*>({da.name}));")
    destroy_body = "\n".join(destroy_lines) if destroy_lines else "        // nothing to free"

    # Build data load body (main function: read binary files, cudaMalloc, cudaMemcpy)
    data_load_lines = []
    for da in all_data:
        if da.dtype == "float":
            data_load_lines.append(f'    snprintf(path, sizeof(path), "%s/{da.name}.bin", data_dir);')
            data_load_lines.append(f'    float* h_{da.name} = read_binary_floats(path, {da.size});')
            data_load_lines.append(f'    float* d_{da.name}; CUDA_CHECK(cudaMalloc(&d_{da.name}, sizeof(float) * {da.size}));')
            data_load_lines.append(f'    CUDA_CHECK(cudaMemcpy(d_{da.name}, h_{da.name}, sizeof(float) * {da.size}, cudaMemcpyHostToDevice));')
            data_load_lines.append(f'    prob.{da.name} = d_{da.name};')
            data_load_lines.append(f'    delete[] h_{da.name};')
        else:
            data_load_lines.append(f'    snprintf(path, sizeof(path), "%s/{da.name}.bin", data_dir);')
            data_load_lines.append(f'    int* h_{da.name} = read_binary_ints(path, {da.size});')
            data_load_lines.append(f'    int* d_{da.name}; CUDA_CHECK(cudaMalloc(&d_{da.name}, sizeof(int) * {da.size}));')
            data_load_lines.append(f'    CUDA_CHECK(cudaMemcpy(d_{da.name}, h_{da.name}, sizeof(int) * {da.size}, cudaMemcpyHostToDevice));')
            data_load_lines.append(f'    prob.{da.name} = d_{da.name};')
            data_load_lines.append(f'    delete[] h_{da.name};')
    data_load_body = "\n".join(data_load_lines) if data_load_lines else "    // no data to load"

    # Build solver config
    config_lines = [
        f"    cfg.pop_size = {pop_size};",
        f"    cfg.max_gen = {max_gen};",
        f"    cfg.seed = {seed};",
        f"    cfg.verbose = {'true' if verbose else 'false'};",
        f"    cfg.use_aos = {'true' if use_aos else 'false'};",
    ]
    if time_limit > 0:
        config_lines.append(f"    cfg.time_limit_sec = {time_limit}f;")
    if sa_temp_init > 0:
        config_lines.append(f"    cfg.sa_temp_init = {sa_temp_init}f;")
    solver_config = "\n".join(config_lines)

    dim2_default = n if row_mode.lower() == "single" else 0
    if total_elements == 0 and row_mode.lower() == "partition":
        total_elements = n

    # Process custom operators
    custom_op_defines_block = ""
    custom_op_switch = ""
    custom_op_registry = ""
    if custom_operators:
        from cugenopt.operators import generate_custom_operator_cuda
        switch_block, registry_block, filtered = generate_custom_operator_cuda(
            custom_operators, encoding
        )
        if filtered:
            custom_op_switch = switch_block
            custom_op_registry = registry_block
            custom_op_defines_block = _build_custom_op_defines(switch_block, registry_block)

    # Fill template
    source = template
    replacements = {
        "{{D1}}": str(dim1),
        "{{D2}}": str(dim2),
        "{{DATA_FIELDS}}": data_fields,
        "{{OBJ_DEFS}}": obj_defs,
        "{{COMPUTE_OBJ}}": compute_obj,
        "{{COMPUTE_PENALTY}}": compute_penalty,
        "{{ENCODING}}": _build_encoding_str(encoding),
        "{{DIM1}}": str(dim1),
        "{{DIM2_DEFAULT}}": str(dim2_default),
        "{{ROW_MODE}}": _build_row_mode_str(row_mode),
        "{{TOTAL_ELEMENTS}}": str(total_elements),
        "{{CROSS_ROW_PROB}}": f"{cross_row_prob}f",
        "{{PERM_REPEAT_COUNT}}": str(perm_repeat_count),
        "{{VALUE_LOWER}}": str(value_lower),
        "{{VALUE_UPPER}}": str(value_upper),
        "{{SHARED_MEM_EXPR}}": shared_mem_expr,
        "{{LOAD_SHARED_BODY}}": load_shared_body,
        "{{DESTROY_BODY}}": destroy_body,
        "{{DATA_LOAD_BODY}}": data_load_body,
        "{{SOLVER_CONFIG}}": solver_config,
        "{{NUM_OBJ}}": str(len(objectives)),
        "{{CUSTOM_OP_DEFINES}}": custom_op_defines_block,
        "{{CUSTOM_OP_SWITCH}}": custom_op_switch,
        "{{CUSTOM_OP_REGISTRY}}": custom_op_registry,
    }
    for key, val in replacements.items():
        source = source.replace(key, val)

    # Hash for caching
    h = _content_hash(source)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cu_path = _CACHE_DIR / f"custom_{h}.cu"
    exe_suffix = ".exe" if _is_windows() else ""
    exe_path = _CACHE_DIR / f"custom_{h}{exe_suffix}"

    # Compile if needed
    if not exe_path.exists():
        cu_path.write_text(source)

        if cuda_arch is None:
            cuda_arch = _detect_cuda_arch()
        nvcc = _find_nvcc()

        cmd = [
            nvcc, "-O2", "-std=c++17", "--extended-lambda", "--expt-relaxed-constexpr",
            f"-arch={cuda_arch}",
            f"-I{fw_root}",
            f"-I{fw_root / 'core'}",
            f"-I{fw_root / 'problems'}",
            str(cu_path), "-o", str(exe_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            if custom_operators and len(custom_operators) > 0:
                source, exe_path, cu_path = _fallback_compile_without_bad_ops(
                    custom_operators, encoding, template, replacements,
                    cmd[:-3], fw_root, proc.stderr,
                )
            else:
                raise CuGenOptCompileError(proc.stderr, str(cu_path))

    # Write data to temp dir
    with tempfile.TemporaryDirectory(prefix="cugenopt_") as tmpdir:
        # Write n
        with open(os.path.join(tmpdir, "n.bin"), "wb") as f:
            f.write(struct.pack("i", n))

        # Write data arrays
        for da in all_data:
            bin_path = os.path.join(tmpdir, f"{da.name}.bin")
            if da.dtype == "float":
                _write_binary_float(bin_path, da.data)
            else:
                _write_binary_int(bin_path, da.data)

        # Run
        proc = subprocess.run(
            [str(exe_path), tmpdir],
            capture_output=True, text=True, timeout=max(300, time_limit * 3 + 60)
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Execution failed:\n{proc.stderr}")

        # Parse JSON output (find the last line that starts with '{')
        output_lines = proc.stdout.strip().split("\n")
        json_line = None
        for line in reversed(output_lines):
            line = line.strip()
            if line.startswith("{"):
                json_line = line
                break

        if json_line is None:
            raise RuntimeError(
                f"No JSON output found.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        result = json.loads(json_line)

        # Convert solution lists to numpy arrays
        if "solution" in result:
            result["solution"] = [np.array(row, dtype=np.int32) for row in result["solution"]]

        return result


def clear_cache():
    """Remove all cached JIT compilations."""
    if _CACHE_DIR.exists():
        shutil.rmtree(_CACHE_DIR)
        print(f"Cleared JIT cache: {_CACHE_DIR}")
