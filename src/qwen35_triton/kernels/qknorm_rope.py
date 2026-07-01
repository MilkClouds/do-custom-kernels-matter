"""Fused per-head QK-norm + partial RoPE Triton kernel for Qwen3.5 attention.

``Qwen3_5Attention`` normalizes every query/key head with an RMSNorm over
``head_dim`` and then applies *partial* rotary embeddings (Qwen3.5 rotates only
the first ``rotary_dim = head_dim * partial_rotary_factor`` channels). In eager
mode that is one RMSNorm plus ~10 elementwise ops (slice, rotate_half, mul,
add, cat) per Q and per K. This kernel folds the norm and the partial RoPE into
a single launch per tensor.

The kernel is fully config-driven: ``head_dim``, ``rotary_dim`` and the
sequence length are read from the tensor arguments — nothing is pinned to a
model size. It handles any ``rotary_dim <= head_dim`` (including the no-pass
case ``rotary_dim == head_dim``) and arbitrary batch / heads / seq length.

Numerical contract — matched to ``q_norm`` + ``apply_rotary_pos_emb``::

    # RMSNorm over head_dim, Qwen3.5 (1 + weight) scaling:
    xn = x.float() * rsqrt(mean(x.float()**2) + eps) * (1 + weight.float())
    xn = xn.to(input_dtype)
    # partial RoPE on the first rotary_dim channels, rotate_half layout:
    half = rotary_dim // 2
    rot[:half]  = xn[:half] * cos[:half]  - xn[half:rotary_dim] * sin[:half]
    rot[half:]  = xn[half:] * cos[half:]  + xn[:half]           * sin[half:]
    out = concat(rot, xn[rotary_dim:])

``cos`` / ``sin`` carry the full ``rotary_dim`` width (HF builds them as
``cat((freqs, freqs))`` so the two halves are equal, but the kernel does not
rely on that and indexes the full width).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import TORCH_TO_TL, triton_block_size


@triton.jit
def _qknorm_rope_kernel(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_x_b,
    stride_x_h,
    stride_x_s,
    stride_x_d,
    stride_out_b,
    stride_out_h,
    stride_out_s,
    stride_out_d,
    stride_cos_b,
    stride_cos_s,
    stride_sin_b,
    stride_sin_s,
    n_heads,
    seq_len,
    head_dim,
    rotary_dim,
    half_rot,
    eps,
    BLOCK_D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One program per (batch, head, seq) triple: RMSNorm(head_dim) then partial RoPE."""
    pid = tl.program_id(0)
    s = pid % seq_len
    h = (pid // seq_len) % n_heads
    b = pid // (seq_len * n_heads)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim

    x_row = x_ptr + b * stride_x_b + h * stride_x_h + s * stride_x_s
    out_row = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

    x = tl.load(x_row + offs * stride_x_d, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm over head_dim, Qwen3.5 (1 + weight) scaling. Masked lanes are 0
    # and drop out of the sum.
    var = tl.sum(x * x, axis=0) / head_dim
    rrms = tl.rsqrt(var + eps)
    # Round the normed value back to the input dtype before RoPE, matching the
    # HF order (q_norm returns input-dtype, then apply_rotary_pos_emb runs).
    xn = (x * rrms * (1.0 + w)).to(OUT_DTYPE).to(tl.float32)

    # Partial RoPE: rotate the first rotary_dim channels, pass the rest through.
    # rotate_half layout splits the rotary span at half_rot.
    is_rot = offs < rotary_dim
    is_low = offs < half_rot

    # Partner index: low half pairs with offs+half_rot, high half with offs-half_rot.
    partner = tl.where(is_low, offs + half_rot, offs - half_rot)
    x_partner = tl.load(x_row + partner * stride_x_d, mask=is_rot, other=0.0).to(tl.float32)
    # Re-normalize the partner channel with its own weight (same rrms).
    w_partner = tl.load(w_ptr + partner, mask=is_rot, other=0.0).to(tl.float32)
    xn_partner = (x_partner * rrms * (1.0 + w_partner)).to(OUT_DTYPE).to(tl.float32)

    # cos/sin are indexed by the channel within the rotary span.
    cos = tl.load(cos_ptr + b * stride_cos_b + s * stride_cos_s + offs, mask=is_rot, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + b * stride_sin_b + s * stride_sin_s + offs, mask=is_rot, other=0.0).to(tl.float32)

    # rotate_half sign: low half subtracts the partner, high half adds it.
    rot_sign = tl.where(is_low, -1.0, 1.0)
    rotated = xn * cos + rot_sign * xn_partner * sin

    out = tl.where(is_rot, rotated, xn)
    tl.store(out_row + offs * stride_out_d, out.to(OUT_DTYPE), mask=mask)


def _rope_table_strides(t: torch.Tensor, batch: int, seq_len: int) -> tuple[int, int]:
    """Resolve (batch_stride, seq_stride) for a cos/sin table.

    Accepts the table as ``[B, S, rotary_dim]`` or as a broadcastable
    ``[S, rotary_dim]`` / ``[1, S, rotary_dim]``; in the broadcast case the
    batch stride is 0 so every batch row reads the shared table.
    """
    if t.ndim == 3:
        b_stride = t.stride(0) if t.shape[0] == batch else 0
        s_stride = t.stride(1)
    elif t.ndim == 2:
        b_stride = 0
        s_stride = t.stride(0)
    else:
        raise ValueError(f"cos/sin table must be 2D or 3D, got shape {tuple(t.shape)}")
    if t.shape[-2] not in (seq_len, 1):
        raise ValueError(f"cos/sin seq dim {t.shape[-2]} incompatible with seq_len {seq_len}")
    if t.shape[-2] == 1:
        s_stride = 0
    return b_stride, s_stride


def fused_qknorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused per-head RMSNorm + partial RoPE for a Qwen3.5 Q or K tensor.

    Args:
        x: ``[B, num_heads, S, head_dim]`` query or key states (pre-norm).
        weight: ``[head_dim]`` RMSNorm weight (``(1 + weight)`` scaling).
        cos: ``[B, S, rotary_dim]`` (or broadcastable ``[S, rotary_dim]``)
            cosine table. ``rotary_dim`` is read from this tensor.
        sin: ``[B, S, rotary_dim]`` sine table, same layout as ``cos``.
        eps: RMSNorm epsilon.

    Returns:
        ``[B, num_heads, S, head_dim]`` normed-and-rotated tensor in ``x``'s dtype.
    """
    if x.ndim != 4:
        raise ValueError(f"x must be [B, num_heads, S, head_dim], got {tuple(x.shape)}")
    batch, n_heads, seq_len, head_dim = x.shape
    if weight.shape[-1] != head_dim:
        raise ValueError(f"weight dim {weight.shape[-1]} != head_dim {head_dim}")

    rotary_dim = cos.shape[-1]
    if sin.shape[-1] != rotary_dim:
        raise ValueError(f"cos rotary_dim {rotary_dim} != sin rotary_dim {sin.shape[-1]}")
    if rotary_dim > head_dim:
        raise ValueError(f"rotary_dim {rotary_dim} exceeds head_dim {head_dim}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim {rotary_dim} must be even for rotate_half")

    cos_b, cos_s = _rope_table_strides(cos, batch, seq_len)
    sin_b, sin_s = _rope_table_strides(sin, batch, seq_len)

    x = x.contiguous()
    out = torch.empty_like(x)
    block_d = triton_block_size(head_dim)
    grid = (batch * n_heads * seq_len,)

    _qknorm_rope_kernel[grid](
        x,
        weight.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        cos_b,
        cos_s,
        sin_b,
        sin_s,
        n_heads,
        seq_len,
        head_dim,
        rotary_dim,
        rotary_dim // 2,
        eps,
        BLOCK_D=block_d,
        OUT_DTYPE=TORCH_TO_TL[x.dtype],
    )
    return out
