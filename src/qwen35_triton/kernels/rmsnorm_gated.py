"""Fused gated RMSNorm Triton kernel for the Qwen3.5 DeltaNet output norm.

The DeltaNet (linear-attention) block normalizes its core attention output and
gates it with a SiLU of a separate projection. This kernel replaces the ~10-op
HF ``Qwen3_5RMSNormGated.forward`` with a single launch.

Numerical contract — matched to ``Qwen3_5RMSNormGated`` exactly::

    h  = hidden.float()
    h  = h * rsqrt(mean(h**2) + eps)
    h  = weight * h.to(input_dtype)         # cast back BEFORE the weight mul
    h  = h * F.silu(gate.float())           # SiLU evaluated in fp32
    return h.to(input_dtype)

Two details that differ from the plain attention RMSNorm and must not be
"simplified" away:

- the scale is plain ``weight`` (initialized to ones), **not** ``1 + weight``;
- there is a mid-computation round-trip to the input dtype right after the
  rsqrt scaling and before the weight multiply. The kernel reproduces that
  rounding so bf16 results match the reference bit-for-bit-ish (within the
  documented tolerance).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import TORCH_TO_TL, triton_block_size


@triton.jit
def _rmsnorm_gated_kernel(
    h_ptr,
    g_ptr,
    w_ptr,
    y_ptr,
    stride_h_row,
    stride_g_row,
    stride_y_row,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One program per row: y = (weight * round(rmsnorm(h))) * silu(g)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols

    h = tl.load(h_ptr + row * stride_h_row + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * stride_g_row + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(h * h, axis=0) / n_cols
    rrms = tl.rsqrt(var + eps)
    # HF rounds the rsqrt-scaled hidden state back to the input dtype *before*
    # the weight multiply. Reproduce that round-trip so bf16 paths match.
    h_normed = (h * rrms).to(OUT_DTYPE).to(tl.float32)
    h_scaled = w * h_normed

    # SiLU(gate) evaluated in fp32, as in the reference.
    silu_g = g * tl.sigmoid(g)
    y = h_scaled * silu_g

    tl.store(y_ptr + row * stride_y_row + offs, y.to(OUT_DTYPE), mask=mask)


def rmsnorm_gated(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused gated RMSNorm — equivalent to ``Qwen3_5RMSNormGated``.

    Args:
        hidden: ``[..., N]`` pre-norm hidden state.
        gate: ``[..., N]`` gate tensor, SiLU-activated and multiplied in.
        weight: ``[N]`` norm weight (plain ``weight`` scaling, not ``1 + w``).
        eps: variance epsilon.

    Returns:
        ``[..., N]`` result in ``hidden``'s dtype.
    """
    if hidden.shape != gate.shape:
        raise ValueError(f"hidden shape {hidden.shape} != gate shape {gate.shape}")
    if hidden.shape[-1] != weight.shape[-1]:
        raise ValueError(f"weight dim {weight.shape[-1]} != last dim {hidden.shape[-1]}")

    orig_shape = hidden.shape
    n_cols = orig_shape[-1]
    h_2d = hidden.reshape(-1, n_cols).contiguous()
    g_2d = gate.reshape(-1, n_cols).contiguous()
    n_rows = h_2d.shape[0]

    y = torch.empty_like(h_2d)
    block_n = triton_block_size(n_cols)
    _rmsnorm_gated_kernel[(n_rows,)](
        h_2d,
        g_2d,
        weight.contiguous(),
        y,
        h_2d.stride(0),
        g_2d.stride(0),
        y.stride(0),
        n_cols,
        eps,
        BLOCK_N=block_n,
        OUT_DTYPE=TORCH_TO_TL[hidden.dtype],
    )
    return y.reshape(orig_shape)
