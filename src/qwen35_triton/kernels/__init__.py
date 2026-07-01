"""Triton kernels for the hot Qwen3.5 ops.

Each kernel collapses a multi-op eager pattern into a single Triton launch and
is config-driven: dimensions are read from the tensor arguments at call time,
so one code path is correct for the dense Qwen3.5 family (0.8B-27B).
"""

from .mlp import fused_silu_mul
from .qknorm_rope import fused_qknorm_rope
from .rmsnorm import fused_residual_rmsnorm, rmsnorm
from .rmsnorm_gated import rmsnorm_gated

__all__ = [
    "fused_qknorm_rope",
    "fused_residual_rmsnorm",
    "fused_silu_mul",
    "rmsnorm",
    "rmsnorm_gated",
]
