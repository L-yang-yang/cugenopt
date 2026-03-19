"""
Custom operator registration for cuGenOpt.

Allows users to inject problem-specific CUDA search operators into the
JIT-compiled solver. Custom operators participate in AOS weight competition
alongside built-in operators.

Two entry points (same underlying mechanism):
  1. Python users: pass custom_operators=[CustomOperator(...)] to solve_custom()
  2. CUDA developers: write the same code body in .cuh and call register_custom_operators()

Operator contract:
  The code body has access to:
    - sol: Solution reference (sol.data[row][col], sol.dim2_sizes[row])
    - rng: curandState* for random number generation
    - dim1, encoding, val_lb, val_ub: problem parameters
    - prob: const CustomProblem* (access data via prob->d_dist, prob->_n, etc.)
  Must return bool (true if solution was modified, false for no-op).
  Available primitives: ops::perm_swap, ops::perm_reverse, ops::perm_insert,
  ops::bin_flip, ops::rand_int, etc.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from cugenopt.validation import validate_cuda_snippet, CuGenOptValidationError


@dataclass
class CustomOperator:
    """A user-defined search operator.

    Args:
        name: Human-readable name (used in logs and AOS stats).
        code: CUDA code body that modifies `sol` in-place.
            Available variables:
              - sol: Solution reference (sol.data[row][col], sol.dim2_sizes[row])
              - rng: curandState* for random number generation
              - dim1: number of active rows
              - prob: const CustomProblem* — access problem data via prob->field
                (e.g. prob->d_dist, prob->_n, prob->d_weights)
            Must return bool (true if sol was modified).
        encoding: Which encoding this operator targets ("permutation", "binary",
            "integer", or "any"). Operators are only active when the problem
            encoding matches.
        initial_weight: Starting AOS weight (relative, will be normalized).
            Higher = more likely to be sampled initially.
        weight_cap: Maximum AOS weight after normalization (0 = use global cap).
    """
    name: str
    code: str
    encoding: str = "any"
    initial_weight: float = 0.5
    weight_cap: float = 0.0

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise CuGenOptValidationError("CustomOperator name cannot be empty")
        self.code = validate_cuda_snippet(self.code, f"operator '{self.name}'")
        valid_enc = {"permutation", "binary", "integer", "any"}
        if self.encoding.lower() not in valid_enc:
            raise CuGenOptValidationError(
                f"CustomOperator encoding must be one of {valid_enc}, "
                f"got '{self.encoding}'"
            )
        if self.initial_weight <= 0:
            raise CuGenOptValidationError(
                f"CustomOperator initial_weight must be > 0, got {self.initial_weight}"
            )


# SeqID range for custom operators: 100..123 (MAX_SEQ=24 custom slots)
CUSTOM_SEQ_ID_BASE = 100


def generate_custom_operator_cuda(
    operators: List[CustomOperator],
    problem_encoding: str,
) -> tuple:
    """Generate CUDA code to inject custom operators into execute_custom_op.

    Returns:
        (switch_block, registry_block, filtered_operators):
          - switch_block: CUDA switch cases for {{CUSTOM_OP_SWITCH}}
          - registry_block: add() calls for {{CUSTOM_OP_REGISTRY}}
          - filtered_operators: list of operators that matched the encoding
        All empty strings / empty list if no operators match.
    """
    if not operators:
        return "", "", ""

    filtered = _filter_by_encoding(operators, problem_encoding)
    if not filtered:
        return "", "", ""

    switch_cases = []
    registry_adds = []
    for i, op in enumerate(filtered):
        seq_id = CUSTOM_SEQ_ID_BASE + i
        switch_cases.append(_generate_switch_case(seq_id, op))
        registry_adds.append(
            f"    add({seq_id}, {op.initial_weight}f, {op.weight_cap}f);  "
            f"// custom: {op.name}"
        )

    switch_block = "\n".join(switch_cases)
    registry_block = "\n".join(registry_adds)

    return switch_block, registry_block, filtered


def _filter_by_encoding(
    operators: List[CustomOperator],
    problem_encoding: str,
) -> List[CustomOperator]:
    """Filter operators compatible with the problem encoding."""
    enc = problem_encoding.lower()
    return [
        op for op in operators
        if op.encoding.lower() == "any" or op.encoding.lower() == enc
    ]


def _generate_switch_case(seq_id: int, op: CustomOperator) -> str:
    """Generate a single switch case for execute_sequence."""
    return f"""\
        case {seq_id}: {{ // custom: {op.name}
{_indent(op.code, 12)}
        }}"""


def _indent(text: str, spaces: int) -> str:
    """Indent each line of text."""
    prefix = " " * spaces
    lines = text.strip().split("\n")
    return "\n".join(prefix + line for line in lines)
