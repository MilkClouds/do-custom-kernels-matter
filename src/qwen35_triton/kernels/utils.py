"""Shared helpers for the Qwen3.5 Triton kernels.

Kept deliberately small: a torch -> Triton dtype map (so kernels can store in
the *input* dtype rather than unconditionally bf16) and a block-size helper.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Triton compile-time dtype for each torch dtype the kernels accept. Passed as a
# ``tl.constexpr`` so a kernel writes its output in the caller's dtype.
TORCH_TO_TL: dict[torch.dtype, tl.dtype] = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

# Triton's static shared-memory / register budget caps a single ``tl.arange``
# block. A power-of-two row width past this needs a tiled kernel; the
# single-block norm kernels here assert against it instead of silently
# truncating. 64K elements covers every Qwen3.5 row width (max is the 27B
# intermediate size, 17408).
MAX_BLOCK_N = 65536


def triton_block_size(n: int) -> int:
    """Next power of two >= ``n``, validated against the single-block ceiling.

    The single-row norm/elementwise kernels load a whole row into one block, so
    the row width must round up to a power of two within :data:`MAX_BLOCK_N`.
    """
    if n <= 0:
        raise ValueError(f"row width must be positive, got {n}")
    block = int(triton.next_power_of_2(n))
    if block > MAX_BLOCK_N:
        raise ValueError(
            f"row width {n} (block {block}) exceeds the single-block limit "
            f"{MAX_BLOCK_N}; a tiled kernel is required for this shape"
        )
    return block
