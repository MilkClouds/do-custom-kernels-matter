"""Fused RMSNorm and residual-add + RMSNorm Triton kernels for Qwen3.5.

Two kernels, both config-driven (the row width ``N`` is read from the input
tensor — nothing is hardcoded to a model size):

- :func:`rmsnorm` — replaces the 7-op HF ``Qwen3_5RMSNorm.forward`` (cast,
  square, mean, rsqrt, mul, mul, cast) with one launch.
- :func:`fused_residual_rmsnorm` — fuses the layer-boundary ``residual + x``
  add into the same launch as the norm, and returns *both* the updated
  residual and the normed output. The HF decoder layer does this add + norm
  twice per layer; folding the add in removes an elementwise launch each time.

Numerical contract — matched to ``Qwen3_5RMSNorm`` exactly::

    output = x.float() * rsqrt(mean(x.float()**2) + eps)
    output = output * (1.0 + weight.float())
    return output.type_as(x)

The ``(1.0 + weight)`` scaling is the Qwen3.5 RMSNorm convention (weight is
initialized to zeros). Variance and the scale multiply happen in fp32; the
result is cast back to the *input* dtype, not unconditionally to bf16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import TORCH_TO_TL, triton_block_size


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x_row,
    stride_y_row,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One program per row: y = (x * rrms) * (1 + w), rrms = rsqrt(mean(x^2)+eps)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols

    x = tl.load(x_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / n_cols
    rrms = tl.rsqrt(var + eps)
    y = x * rrms * (1.0 + w)

    tl.store(y_ptr + row * stride_y_row + offs, y.to(OUT_DTYPE), mask=mask)


@triton.jit
def _fused_residual_rmsnorm_kernel(
    r_ptr,
    x_ptr,
    w_ptr,
    new_r_ptr,
    y_ptr,
    stride_r_row,
    stride_x_row,
    stride_new_r_row,
    stride_y_row,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One program per row: new_r = r + x; y = rmsnorm(new_r) * (1 + w)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols

    r = tl.load(r_ptr + row * stride_r_row + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Residual add in fp32. The new residual is the *un-normed* sum and is
    # stored in the input dtype so the next layer's residual path is unchanged.
    h = r + x

    var = tl.sum(h * h, axis=0) / n_cols
    rrms = tl.rsqrt(var + eps)
    y = h * rrms * (1.0 + w)

    tl.store(new_r_ptr + row * stride_new_r_row + offs, h.to(OUT_DTYPE), mask=mask)
    tl.store(y_ptr + row * stride_y_row + offs, y.to(OUT_DTYPE), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused Qwen3.5 RMSNorm — numerically equivalent to ``Qwen3_5RMSNorm``.

    Args:
        x: ``[..., N]`` input. Any floating dtype; the result keeps this dtype.
        weight: ``[N]`` norm weight. The kernel applies ``(1 + weight)`` scaling.
        eps: variance epsilon.

    Returns:
        ``[..., N]`` normalized tensor in ``x``'s dtype.
    """
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(f"weight dim {weight.shape[-1]} != x last dim {x.shape[-1]}")

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    n_rows, n_cols = x_2d.shape
    y = torch.empty_like(x_2d)

    block_n = triton_block_size(n_cols)
    _rmsnorm_kernel[(n_rows,)](
        x_2d,
        weight.contiguous(),
        y,
        x_2d.stride(0),
        y.stride(0),
        n_cols,
        eps,
        BLOCK_N=block_n,
        OUT_DTYPE=TORCH_TO_TL[x.dtype],
    )
    return y.reshape(orig_shape)


def fused_residual_rmsnorm(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``residual + x`` then RMSNorm, in a single launch.

    Computes ``new_residual = residual + x`` and ``normed =
    rmsnorm(new_residual, weight, eps)``, equivalent to the HF decoder-layer
    pattern ``hidden = residual + x; normed = post_attention_layernorm(hidden)``.

    Args:
        residual: ``[..., N]`` residual-stream tensor.
        x: ``[..., N]`` block output to add into the residual.
        weight: ``[N]`` norm weight (``(1 + weight)`` scaling applied).
        eps: variance epsilon.

    Returns:
        ``(new_residual, normed)`` — both ``[..., N]`` in ``residual``'s dtype.
        ``new_residual`` is the raw (un-normed) sum.
    """
    if residual.shape != x.shape:
        raise ValueError(f"residual shape {residual.shape} != x shape {x.shape}")
    if residual.shape[-1] != weight.shape[-1]:
        raise ValueError(f"weight dim {weight.shape[-1]} != last dim {residual.shape[-1]}")

    orig_shape = residual.shape
    n_cols = orig_shape[-1]
    r_2d = residual.reshape(-1, n_cols).contiguous()
    x_2d = x.reshape(-1, n_cols).contiguous()
    n_rows = r_2d.shape[0]

    new_r = torch.empty_like(r_2d)
    normed = torch.empty_like(r_2d)

    block_n = triton_block_size(n_cols)
    _fused_residual_rmsnorm_kernel[(n_rows,)](
        r_2d,
        x_2d,
        weight.contiguous(),
        new_r,
        normed,
        r_2d.stride(0),
        x_2d.stride(0),
        new_r.stride(0),
        normed.stride(0),
        n_cols,
        eps,
        BLOCK_N=block_n,
        OUT_DTYPE=TORCH_TO_TL[residual.dtype],
    )
    return new_r.reshape(orig_shape), normed.reshape(orig_shape)
