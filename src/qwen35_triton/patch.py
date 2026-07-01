"""``patch_model`` — swap a HF Qwen3.5 model's hot ops onto the Triton kernels.

The patch is monkey-patch based and operates on the upstream module *classes*
(``Qwen3_5RMSNorm`` etc.), so a single call covers every layer of a loaded
model regardless of size. Dimensions are read per-module from the module's own
attributes / config — nothing is hardcoded to 0.8B through 27B.

Three forwards are patched unconditionally — each is a clean drop-in:

- ``Qwen3_5RMSNorm.forward`` -> :func:`rmsnorm`. This also covers the attention
  ``q_norm`` / ``k_norm`` (they are ``Qwen3_5RMSNorm`` instances), so per-head
  QK-norm is accelerated without touching the attention forward.
- ``Qwen3_5RMSNormGated.forward`` -> :func:`rmsnorm_gated` (DeltaNet output
  norm).
- ``Qwen3_5MLP.forward`` -> the GEMMs stay on cuBLAS, the ``silu(gate) * up``
  elementwise core goes to :func:`fused_silu_mul`.

One forward is patched only when ``fuse_qk_rope=True`` (opt-in):

- ``Qwen3_5Attention.forward`` -> a copy that fuses the QK-norm and the partial
  RoPE into :func:`fused_qknorm_rope`. This replaces the whole attention
  forward, so it is version-guarded: it only installs against the transformers
  release it was written for, and raises otherwise rather than risk silent
  drift. The default-on patches above already accelerate QK-norm via the
  ``Qwen3_5RMSNorm`` swap; ``fuse_qk_rope`` only adds the norm+RoPE fusion.

All patches are idempotent and reversible with :func:`unpatch_model`.
"""

from __future__ import annotations

import torch

from .kernels import fused_qknorm_rope, fused_silu_mul, rmsnorm, rmsnorm_gated

# transformers releases whose Qwen3_5Attention.forward is byte-identical to the
# copy used by the fuse_qk_rope path below. Verified against 5.8.0 and 5.9.0 —
# the forward did not change between them. The unconditional class-forward swaps
# (the default patch_model path) do not depend on this.
_SUPPORTED_TRANSFORMERS = ("5.8.0", "5.9.0")

# Saved originals, for unpatch. Keyed by a short label.
_ORIGINALS: dict[str, object] = {}


def _qwen35_module():
    """Import the upstream Qwen3.5 modeling module, or raise a clear error."""
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError as exc:  # pragma: no cover - depends on the installed transformers
        raise ImportError(
            "qwen35-triton requires a transformers build that ships the Qwen3.5 "
            "model (transformers>=5.6). Could not import "
            "transformers.models.qwen3_5.modeling_qwen3_5."
        ) from exc
    return modeling_qwen3_5


# ── patched forwards ─────────────────────────────────────────────────────────


def _patched_rmsnorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in for ``Qwen3_5RMSNorm.forward``.

    ``Qwen3_5RMSNorm`` stores the variance epsilon as ``self.eps``; the kernel
    applies the ``(1 + weight)`` scaling that the module uses.
    """
    return rmsnorm(x, self.weight, self.eps)


def _patched_rmsnorm_gated_forward(
    self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None
) -> torch.Tensor:
    """Drop-in for ``Qwen3_5RMSNormGated.forward``.

    ``Qwen3_5RMSNormGated`` stores the epsilon as ``self.variance_epsilon`` and
    uses plain ``weight`` scaling (no ``1 +``). Falls back to the eager path if
    called without a gate, which the upstream signature permits.
    """
    if gate is None:
        # No gate -> plain RMSNorm with plain-weight scaling (not 1 + weight).
        dt = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight * h.to(dt)).to(dt)
    return rmsnorm_gated(hidden_states, gate, self.weight, self.variance_epsilon)


def _patched_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in for ``Qwen3_5MLP.forward``.

    The gate / up / down projections stay on cuBLAS (already well served); the
    ``silu(gate) * up`` elementwise step is fused into one Triton launch.
    """
    return self.down_proj(fused_silu_mul(self.gate_proj(x), self.up_proj(x)))


def _make_patched_attention_forward(modeling):
    """Build a ``Qwen3_5Attention.forward`` copy with fused QK-norm + RoPE.

    Mirrors the upstream forward op-for-op, except the two ``q_norm`` /
    ``k_norm`` RMSNorms and the ``apply_rotary_pos_emb`` call are replaced by a
    single :func:`fused_qknorm_rope` launch per Q and per K.
    """
    eager_attention_forward = modeling.eager_attention_forward
    all_attention_functions = modeling.ALL_ATTENTION_FUNCTIONS

    def _patched_attention_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        cos, sin = position_embeddings

        # Fused QK-norm + partial RoPE. q_norm/k_norm are Qwen3_5RMSNorm over
        # head_dim with (1 + weight) scaling, matching fused_qknorm_rope. The
        # tensors are [B, H, S, head_dim] after the transpose, which is the
        # layout the kernel and apply_rotary_pos_emb both expect.
        query_states = fused_qknorm_rope(
            query_states.view(hidden_shape).transpose(1, 2), self.q_norm.weight, cos, sin, self.q_norm.eps
        )
        key_states = fused_qknorm_rope(
            self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2),
            self.k_norm.weight,
            cos,
            sin,
            self.k_norm.eps,
        )
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = all_attention_functions.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return _patched_attention_forward


# ── public API ───────────────────────────────────────────────────────────────


def patch_model(model: torch.nn.Module, *, fuse_qk_rope: bool = False) -> torch.nn.Module:
    """Patch a HF Qwen3.5 model in place to use the Triton kernels.

    Swaps the upstream ``Qwen3_5RMSNorm`` / ``Qwen3_5RMSNormGated`` /
    ``Qwen3_5MLP`` ``forward`` methods for fused-Triton equivalents. With
    ``fuse_qk_rope=True`` it additionally replaces ``Qwen3_5Attention.forward``
    to fuse QK-norm with partial RoPE (version-guarded — see the module
    docstring).

    The patch is on the module *classes*, so it covers every layer of every
    loaded Qwen3.5 model in the process. It is idempotent.

    Args:
        model: a loaded HF Qwen3.5 model (or any object — the patch is global).
        fuse_qk_rope: also fuse the attention QK-norm + RoPE. Off by default;
            the default patches already route QK-norm through the Triton
            ``rmsnorm`` kernel via the ``Qwen3_5RMSNorm`` swap.

    Returns:
        The same ``model`` object, for call chaining.
    """
    modeling = _qwen35_module()

    if not getattr(modeling.Qwen3_5RMSNorm, "_qwen35_triton_patched", False):
        _ORIGINALS["rmsnorm"] = modeling.Qwen3_5RMSNorm.forward
        modeling.Qwen3_5RMSNorm.forward = _patched_rmsnorm_forward
        modeling.Qwen3_5RMSNorm._qwen35_triton_patched = True

    if not getattr(modeling.Qwen3_5RMSNormGated, "_qwen35_triton_patched", False):
        _ORIGINALS["rmsnorm_gated"] = modeling.Qwen3_5RMSNormGated.forward
        modeling.Qwen3_5RMSNormGated.forward = _patched_rmsnorm_gated_forward
        modeling.Qwen3_5RMSNormGated._qwen35_triton_patched = True

    if not getattr(modeling.Qwen3_5MLP, "_qwen35_triton_patched", False):
        _ORIGINALS["mlp"] = modeling.Qwen3_5MLP.forward
        modeling.Qwen3_5MLP.forward = _patched_mlp_forward
        modeling.Qwen3_5MLP._qwen35_triton_patched = True

    if fuse_qk_rope and not getattr(modeling.Qwen3_5Attention, "_qwen35_triton_patched", False):
        import transformers

        if transformers.__version__ not in _SUPPORTED_TRANSFORMERS:
            raise RuntimeError(
                f"fuse_qk_rope replaces the whole Qwen3_5Attention.forward, which was "
                f"copied from transformers in {_SUPPORTED_TRANSFORMERS}; "
                f"{transformers.__version__!r} is installed. Re-validate the copy "
                f"against the current upstream forward before enabling fuse_qk_rope."
            )
        _ORIGINALS["attention"] = modeling.Qwen3_5Attention.forward
        modeling.Qwen3_5Attention.forward = _make_patched_attention_forward(modeling)
        modeling.Qwen3_5Attention._qwen35_triton_patched = True

    return model


def unpatch_model(model: torch.nn.Module | None = None) -> None:
    """Restore the original HF Qwen3.5 forwards. Idempotent.

    Args:
        model: unused; accepted so ``patch_model`` / ``unpatch_model`` mirror.
    """
    modeling = _qwen35_module()

    if "rmsnorm" in _ORIGINALS:
        modeling.Qwen3_5RMSNorm.forward = _ORIGINALS.pop("rmsnorm")
        modeling.Qwen3_5RMSNorm._qwen35_triton_patched = False
    if "rmsnorm_gated" in _ORIGINALS:
        modeling.Qwen3_5RMSNormGated.forward = _ORIGINALS.pop("rmsnorm_gated")
        modeling.Qwen3_5RMSNormGated._qwen35_triton_patched = False
    if "mlp" in _ORIGINALS:
        modeling.Qwen3_5MLP.forward = _ORIGINALS.pop("mlp")
        modeling.Qwen3_5MLP._qwen35_triton_patched = False
    if "attention" in _ORIGINALS:
        modeling.Qwen3_5Attention.forward = _ORIGINALS.pop("attention")
        modeling.Qwen3_5Attention._qwen35_triton_patched = False


def is_patched() -> bool:
    """True if the unconditional class-forward patches are currently installed."""
    modeling = _qwen35_module()
    return bool(getattr(modeling.Qwen3_5RMSNorm, "_qwen35_triton_patched", False))
