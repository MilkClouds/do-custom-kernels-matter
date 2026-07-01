"""Fused SiLU-gated MLP elementwise kernel for Qwen3.5.

The HF ``Qwen3_5MLP.forward`` is ``down_proj(act_fn(gate_proj(x)) *
up_proj(x))``. The two input projections and the output projection are plain
GEMMs — cuBLAS already serves those well, and fusing them into Triton would
trade a tuned GEMM for a hand-rolled one. The fusible, launch-bound part is the
``silu(gate) * up`` elementwise step between the projections; this kernel folds
SiLU + multiply into a single launch and skips materializing the intermediate
``silu(gate)`` tensor.

Numerical contract::

    out = silu(gate) * up = (gate * sigmoid(gate)) * up

The activation and multiply run in fp32 internally and the result is cast back
to the input dtype. ``Qwen3_5MLP`` uses ``ACT2FN["silu"]`` (``nn.SiLU``); on a
bf16 input that reference path keeps bf16 intermediates, so the fp32-internal
kernel is *more* accurate than the reference. The correctness gate in
``scripts/qwen35/check_correctness.py`` sizes tolerance against the reference's
own bf16 rounding.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import TORCH_TO_TL


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 512}, num_warps=2),
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_N": 4096}, num_warps=8),
    ],
    key=["n_cols"],
)
@triton.jit
def _silu_mul_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    stride_gate_row,
    stride_up_row,
    stride_out_row,
    n_cols,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """out = silu(gate) * up, tiled over the column dimension."""
    row = tl.program_id(1)
    col_block = tl.program_id(0)
    offs = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_cols

    gate = tl.load(gate_ptr + row * stride_gate_row + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + row * stride_up_row + offs, mask=mask, other=0.0).to(tl.float32)

    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up

    tl.store(out_ptr + row * stride_out_row + offs, out.to(OUT_DTYPE), mask=mask)


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``silu(gate) * up`` — the elementwise core of ``Qwen3_5MLP``.

    Args:
        gate: ``[..., N]`` gate-projection output.
        up: ``[..., N]`` up-projection output. Must match ``gate``'s shape.

    Returns:
        ``[..., N]`` = ``silu(gate) * up`` in ``gate``'s dtype.
    """
    if gate.shape != up.shape:
        raise ValueError(f"gate shape {gate.shape} != up shape {up.shape}")

    orig_shape = gate.shape
    n_cols = orig_shape[-1]
    gate_2d = gate.reshape(-1, n_cols).contiguous()
    up_2d = up.reshape(-1, n_cols).contiguous()
    n_rows = gate_2d.shape[0]

    out = torch.empty_like(gate_2d)

    def grid(meta: dict) -> tuple[int, int]:
        return (triton.cdiv(n_cols, meta["BLOCK_N"]), n_rows)

    _silu_mul_kernel[grid](
        gate_2d,
        up_2d,
        out,
        gate_2d.stride(0),
        up_2d.stride(0),
        out.stride(0),
        n_cols,
        OUT_DTYPE=TORCH_TO_TL[gate.dtype],
    )
    return out.reshape(orig_shape)
