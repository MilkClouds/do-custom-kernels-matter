"""qwen35-triton — config-driven Triton kernels for Qwen3.5 inference.

Qwen3.5 inference in HuggingFace eager mode is CPU-dispatch-bound: a forward
pass spends far longer on the CPU launching kernels than the GPU spends running
them. Each kernel here collapses a multi-op eager pattern into a single Triton
launch, cutting the launch count and peak memory.

Every kernel reads its dimensions from the tensor arguments at call time, so a
single code path is correct for the dense Qwen3.5 family (0.8B-27B) and any
other valid config. :func:`extract_dims` turns a HuggingFace ``Qwen3_5Config``
/ ``Qwen3_5TextConfig`` into an explicit :class:`Qwen35KernelDims` record.

Quick start::

    from transformers import AutoModelForCausalLM
    from qwen35_triton import patch_model

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B")
    patch_model(model)   # Triton kernels now active

Public API:

- :func:`patch_model` / :func:`unpatch_model` / :func:`is_patched` — install or
  remove the kernels on a HF Qwen3.5 model.
- :class:`GraphedDecoder` — CUDA-graphed greedy decode (the end-to-end lever:
  fused kernels shrink each op, the graph removes the per-token launch cost of
  what remains).
- :func:`rmsnorm`, :func:`fused_residual_rmsnorm`, :func:`rmsnorm_gated`,
  :func:`fused_silu_mul`, :func:`fused_qknorm_rope` — the kernels, callable
  directly.
- :func:`extract_dims` / :class:`Qwen35KernelDims` — config -> dimensions.
"""

from .config import Qwen35KernelDims, extract_dims
from .graph import GraphedDecoder
from .kernels import (
    fused_qknorm_rope,
    fused_residual_rmsnorm,
    fused_silu_mul,
    rmsnorm,
    rmsnorm_gated,
)
from .patch import is_patched, patch_model, unpatch_model

__version__ = "0.1.0"

__all__ = [
    "GraphedDecoder",
    "Qwen35KernelDims",
    "extract_dims",
    "fused_qknorm_rope",
    "fused_residual_rmsnorm",
    "fused_silu_mul",
    "is_patched",
    "patch_model",
    "rmsnorm",
    "rmsnorm_gated",
    "unpatch_model",
]
