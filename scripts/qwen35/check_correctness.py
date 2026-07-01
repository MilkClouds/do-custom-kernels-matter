"""Qwen3.5 Triton-kernel correctness gate.

This checks the vendored custom kernels against small PyTorch references. It is
separate from the latency benchmark so benchmark output stays simple, but it
should be run before trusting custom-kernel timing numbers.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F

from qwen35_triton import fused_qknorm_rope, fused_residual_rmsnorm, fused_silu_mul, rmsnorm, rmsnorm_gated

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, *, atol: float, rtol: float) -> None:
    if torch.allclose(got, ref, atol=atol, rtol=rtol):
        return
    diff = (got.float() - ref.float()).abs()
    raise AssertionError(
        f"{name} failed: max_abs={diff.max().item():.6g}, mean_abs={diff.mean().item():.6g}, atol={atol}, rtol={rtol}"
    )


def ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    y = y * (1.0 + weight.float())
    return y.to(x.dtype)


def ref_residual_rmsnorm(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_residual = (residual.float() + x.float()).to(residual.dtype)
    return new_residual, ref_rmsnorm(new_residual, weight, eps)


def ref_rmsnorm_gated(hidden: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    h = hidden.float()
    h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    h = h.to(hidden.dtype).float()
    y = weight.float() * h
    y = y * F.silu(gate.float())
    return y.to(hidden.dtype)


def ref_qknorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    y = ref_rmsnorm(x, weight, eps).float()
    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    rot = y[..., :rotary_dim].clone()
    lo = y[..., :half]
    hi = y[..., half:rotary_dim]
    cos = cos[:, None, :, :].float()
    sin = sin[:, None, :, :].float()
    rot[..., :half] = lo * cos[..., :half] - hi * sin[..., :half]
    rot[..., half:rotary_dim] = hi * cos[..., half:rotary_dim] + lo * sin[..., half:rotary_dim]
    y[..., :rotary_dim] = rot
    return y.to(x.dtype)


def check_kernel(name: str, fn: Callable[[], None]) -> None:
    fn()
    print(f"ok {name}")


def run(device: str, dtype: torch.dtype, seed: int) -> None:
    torch.manual_seed(seed)
    eps = 1e-6
    atol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-5
    rtol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-5

    def check_rmsnorm() -> None:
        x = torch.randn(2, 3, 96, device=device, dtype=dtype)
        w = torch.randn(96, device=device, dtype=dtype) * 0.05
        assert_close("rmsnorm", rmsnorm(x, w, eps), ref_rmsnorm(x, w, eps), atol=atol, rtol=rtol)

    def check_residual_rmsnorm() -> None:
        residual = torch.randn(2, 3, 96, device=device, dtype=dtype)
        x = torch.randn(2, 3, 96, device=device, dtype=dtype)
        w = torch.randn(96, device=device, dtype=dtype) * 0.05
        got_r, got_y = fused_residual_rmsnorm(residual, x, w, eps)
        ref_r, ref_y = ref_residual_rmsnorm(residual, x, w, eps)
        assert_close("fused_residual_rmsnorm.residual", got_r, ref_r, atol=atol, rtol=rtol)
        assert_close("fused_residual_rmsnorm.normed", got_y, ref_y, atol=atol, rtol=rtol)

    def check_rmsnorm_gated() -> None:
        hidden = torch.randn(2, 3, 96, device=device, dtype=dtype)
        gate = torch.randn(2, 3, 96, device=device, dtype=dtype)
        w = torch.randn(96, device=device, dtype=dtype)
        assert_close(
            "rmsnorm_gated",
            rmsnorm_gated(hidden, gate, w, eps),
            ref_rmsnorm_gated(hidden, gate, w, eps),
            atol=atol,
            rtol=rtol,
        )

    def check_silu_mul() -> None:
        gate = torch.randn(2, 3, 128, device=device, dtype=dtype)
        up = torch.randn(2, 3, 128, device=device, dtype=dtype)
        ref = (F.silu(gate.float()) * up.float()).to(dtype)
        assert_close("fused_silu_mul", fused_silu_mul(gate, up), ref, atol=atol, rtol=rtol)

    def check_qknorm_rope() -> None:
        x = torch.randn(2, 4, 5, 64, device=device, dtype=dtype)
        w = torch.randn(64, device=device, dtype=dtype) * 0.05
        cos = torch.randn(2, 5, 48, device=device, dtype=dtype)
        sin = torch.randn(2, 5, 48, device=device, dtype=dtype)
        got = fused_qknorm_rope(x, w, cos, sin, eps)
        ref = ref_qknorm_rope(x, w, cos, sin, eps)
        assert_close("fused_qknorm_rope", got, ref, atol=atol, rtol=rtol)

    check_kernel("rmsnorm", check_rmsnorm)
    check_kernel("fused_residual_rmsnorm", check_residual_rmsnorm)
    check_kernel("rmsnorm_gated", check_rmsnorm_gated)
    check_kernel("fused_silu_mul", check_silu_mul)
    check_kernel("fused_qknorm_rope", check_qknorm_rope)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Triton kernel correctness checks")
    run(args.device, DTYPES[args.dtype], args.seed)


if __name__ == "__main__":
    main()
