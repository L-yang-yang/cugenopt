"""
Input validation and friendly error translation for cuGenOpt.

Two responsibilities:
  1. Validate numpy arrays before JIT compilation (dtype, shape, NaN/Inf, contiguity)
  2. Translate nvcc compilation errors into actionable Python messages
"""

import re
from typing import Dict, Optional, Sequence

import numpy as np


class CuGenOptValidationError(ValueError):
    """Raised when input data fails validation."""
    pass


class CuGenOptCompileError(RuntimeError):
    """Raised when nvcc compilation fails, with a friendly summary."""

    def __init__(self, raw_stderr: str, source_path: str):
        self.raw_stderr = raw_stderr
        self.source_path = source_path
        self.friendly = _translate_nvcc_error(raw_stderr)
        super().__init__(
            f"{self.friendly}\n\n"
            f"[raw nvcc output]\n{_truncate(raw_stderr, 1200)}\n\n"
            f"Source saved at: {source_path}"
        )


# ============================================================
# Array validation
# ============================================================

def validate_array(
    arr: np.ndarray,
    name: str,
    *,
    expected_dtype: Optional[np.dtype] = None,
    expected_ndim: Optional[int] = None,
    expected_shape: Optional[tuple] = None,
    min_size: int = 1,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> np.ndarray:
    """Validate a single numpy array and return a contiguous copy if needed.

    Raises CuGenOptValidationError with a clear message on failure.
    """
    if not isinstance(arr, np.ndarray):
        raise CuGenOptValidationError(
            f"'{name}' must be a numpy array, got {type(arr).__name__}"
        )

    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise CuGenOptValidationError(
            f"'{name}' must be {expected_ndim}D, got {arr.ndim}D with shape {arr.shape}"
        )

    if expected_shape is not None:
        for i, (actual, expect) in enumerate(zip(arr.shape, expected_shape)):
            if expect is not None and actual != expect:
                raise CuGenOptValidationError(
                    f"'{name}' shape mismatch at axis {i}: "
                    f"expected {expected_shape}, got {arr.shape}"
                )

    if arr.size < min_size:
        raise CuGenOptValidationError(
            f"'{name}' is too small: size={arr.size}, minimum={min_size}"
        )

    if expected_dtype is not None:
        arr = np.ascontiguousarray(arr, dtype=expected_dtype)

    if not allow_nan and np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
        nan_count = int(np.isnan(arr).sum())
        raise CuGenOptValidationError(
            f"'{name}' contains {nan_count} NaN value(s). "
            f"Clean your data or set allow_nan=True."
        )

    if not allow_inf and np.issubdtype(arr.dtype, np.floating) and np.isinf(arr).any():
        inf_count = int(np.isinf(arr).sum())
        raise CuGenOptValidationError(
            f"'{name}' contains {inf_count} Inf value(s). "
            f"Clean your data or set allow_inf=True."
        )

    return np.ascontiguousarray(arr)


def validate_square_matrix(arr: np.ndarray, name: str, dtype=np.float32) -> np.ndarray:
    """Validate a square 2D matrix."""
    arr = validate_array(arr, name, expected_ndim=2, expected_dtype=dtype)
    if arr.shape[0] != arr.shape[1]:
        raise CuGenOptValidationError(
            f"'{name}' must be square, got shape {arr.shape}"
        )
    return arr


def validate_1d(arr: np.ndarray, name: str, *, length: Optional[int] = None,
                dtype=np.float32) -> np.ndarray:
    """Validate a 1D array with optional length check."""
    arr = validate_array(arr, name, expected_ndim=1, expected_dtype=dtype)
    if length is not None and arr.shape[0] != length:
        raise CuGenOptValidationError(
            f"'{name}' length mismatch: expected {length}, got {arr.shape[0]}"
        )
    return arr


def validate_data_dict(data: Dict[str, np.ndarray], dtype_tag: str) -> Dict[str, np.ndarray]:
    """Validate a dict of name -> array for compile_and_solve data/int_data."""
    target_dtype = np.float32 if dtype_tag == "float" else np.int32
    validated = {}
    for name, arr in data.items():
        if not isinstance(arr, np.ndarray):
            raise CuGenOptValidationError(
                f"data['{name}'] must be a numpy array, got {type(arr).__name__}"
            )
        arr = validate_array(arr, f"data['{name}']", expected_dtype=target_dtype)
        validated[name] = arr
    return validated


def validate_encoding(encoding: str) -> str:
    """Validate encoding string."""
    valid = {"permutation", "binary", "integer"}
    enc = encoding.lower().strip()
    if enc not in valid:
        raise CuGenOptValidationError(
            f"Unknown encoding '{encoding}'. Must be one of: {', '.join(sorted(valid))}"
        )
    return enc


def validate_positive_int(value, name: str, *, allow_zero: bool = False) -> int:
    """Validate that value is a positive integer."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise CuGenOptValidationError(
            f"'{name}' must be an integer, got {type(value).__name__}: {value!r}"
        )
    if allow_zero and v < 0:
        raise CuGenOptValidationError(f"'{name}' must be >= 0, got {v}")
    if not allow_zero and v < 1:
        raise CuGenOptValidationError(f"'{name}' must be >= 1, got {v}")
    return v


def validate_cuda_snippet(code: str, name: str) -> str:
    """Basic sanity check on a CUDA code snippet."""
    code = code.strip()
    if not code:
        raise CuGenOptValidationError(f"'{name}' CUDA code snippet is empty")

    dangerous = ["system(", "popen(", "exec(", "fork(", "unlink("]
    for d in dangerous:
        if d in code:
            raise CuGenOptValidationError(
                f"'{name}' contains potentially dangerous call: '{d}'"
            )
    return code


# ============================================================
# nvcc error translation
# ============================================================

_NVCC_PATTERNS = [
    (
        re.compile(r"error:\s*identifier\s+\"(\w+)\"\s+is\s+undefined", re.I),
        lambda m: f"Undefined identifier '{m.group(1)}'. "
                  f"Check that all data field names in compute_obj/compute_penalty "
                  f"match the keys in your data dict."
    ),
    (
        re.compile(r"error:\s*expected\s+a\s+\"([^\"]+)\"", re.I),
        lambda m: f"Syntax error: expected '{m.group(1)}'. "
                  f"Check for missing semicolons, braces, or parentheses."
    ),
    (
        re.compile(r"error:\s*no\s+suitable\s+conversion\s+function\s+from\s+\"([^\"]+)\"\s+to\s+\"([^\"]+)\"", re.I),
        lambda m: f"Type mismatch: cannot convert '{m.group(1)}' to '{m.group(2)}'. "
                  f"Ensure you're using the correct types (float/int)."
    ),
    (
        re.compile(r"error:\s*too\s+(?:few|many)\s+arguments", re.I),
        lambda m: f"Wrong number of arguments in a function call. "
                  f"Check the function signature."
    ),
    (
        re.compile(r"error:\s*class\s+\"(\w+)\"\s+has\s+no\s+member\s+\"(\w+)\"", re.I),
        lambda m: f"'{m.group(1)}' has no member '{m.group(2)}'. "
                  f"Available solution members: data[row][col], dim2_sizes[row]."
    ),
    (
        re.compile(r"error:\s*expression\s+must\s+have\s+a\s+constant\s+value", re.I),
        lambda m: f"Non-constant expression where a constant is required. "
                  f"CUDA device code cannot use dynamic allocation; "
                  f"use fixed-size arrays."
    ),
    (
        re.compile(r"ptxas\s+error\s*:\s*Entry\s+function.*uses\s+too\s+much\s+shared\s+data", re.I),
        lambda m: f"Shared memory overflow. Your problem data is too large for GPU "
                  f"shared memory. Try reducing problem size or data arrays."
    ),
    (
        re.compile(r"nvcc\s+fatal\s*:\s*Unsupported\s+gpu\s+architecture\s+'compute_(\d+)'", re.I),
        lambda m: f"GPU architecture sm_{m.group(1)} is not supported by your nvcc. "
                  f"Try specifying cuda_arch='sm_75' or update your CUDA toolkit."
    ),
    (
        re.compile(r"error:\s*return\s+value\s+type\s+does\s+not\s+match", re.I),
        lambda m: f"Return type mismatch. compute_obj must return float. "
                  f"Make sure all code paths return a float value."
    ),
]


def _translate_nvcc_error(stderr: str) -> str:
    """Extract the most relevant error from nvcc output and provide a friendly message."""
    messages = []
    for pattern, formatter in _NVCC_PATTERNS:
        match = pattern.search(stderr)
        if match:
            messages.append(formatter(match))

    if messages:
        header = "nvcc compilation failed. Likely cause(s):\n"
        return header + "\n".join(f"  - {m}" for m in messages)

    error_lines = [
        line.strip() for line in stderr.split("\n")
        if "error" in line.lower() and not line.strip().startswith("#")
    ]
    if error_lines:
        summary = error_lines[0]
        return (
            f"nvcc compilation failed:\n  {summary}\n\n"
            f"Tip: Check your CUDA code snippets for syntax errors. "
            f"Common issues: missing semicolons, undefined variables, "
            f"wrong data field names."
        )

    return (
        "nvcc compilation failed with an unknown error.\n"
        "Check the raw output below for details."
    )


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... ({len(text) - max_len} chars truncated)"
