"""Config-driven dimension extraction for the Qwen3.5 Triton kernels.

Every kernel in this package is shape-agnostic: it reads the dimensions it
needs straight from the tensor arguments at call time. This module exists for
the *integration* side — turning a HuggingFace ``Qwen3_5TextConfig`` (or the
parent ``Qwen3_5Config``) into a flat, explicit record of the dimensions the
kernels care about, so a caller never has to hardcode 0.8B through 27B numbers.

The dense Qwen3.5 presets (0.8B-27B) only differ in ``hidden_size`` /
``intermediate_size`` / ``num_hidden_layers`` / ``num_attention_heads`` /
``linear_num_value_heads``. Pulling them from config keeps a single kernel
path correct for all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _text_config(config: Any) -> Any:
    """Return the text sub-config from either a ``Qwen3_5Config`` or text config.

    ``Qwen3_5Config`` nests the language-model dimensions under ``text_config``;
    a bare ``Qwen3_5TextConfig`` carries them directly.
    """
    return getattr(config, "text_config", None) or config


@dataclass(frozen=True)
class Qwen35KernelDims:
    """Flat record of every dimension the Qwen3.5 Triton kernels consume.

    All fields are derived from the model config — nothing here is hardcoded to
    a particular model size. ``head_dim`` falls back to ``hidden_size //
    num_attention_heads`` when the config omits an explicit ``head_dim``, which
    matches the HF ``Qwen3_5Attention`` constructor.
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    rms_norm_eps: float
    # DeltaNet (linear-attention) dimensions.
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_conv_kernel_dim: int
    conv_dim: int
    # Per-layer token-mixer pattern: "full_attention" or "linear_attention".
    layer_types: tuple[str, ...]

    @property
    def num_key_value_groups(self) -> int:
        """GQA repeat factor for the full-attention layers."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def num_full_attention_layers(self) -> int:
        return sum(t == "full_attention" for t in self.layer_types)

    @property
    def num_linear_attention_layers(self) -> int:
        return sum(t == "linear_attention" for t in self.layer_types)


def extract_dims(config: Any) -> Qwen35KernelDims:
    """Build a :class:`Qwen35KernelDims` from a Qwen3.5 (text or full) config.

    Reads every dimension the kernels need from the config object. Works for
    the dense released family (0.8B / 2B / 4B / 9B / 27B) and any other valid
    Qwen3.5 config; no size is special-cased.
    """
    tc = _text_config(config)

    hidden_size = int(tc.hidden_size)
    num_attention_heads = int(tc.num_attention_heads)
    head_dim = int(getattr(tc, "head_dim", None) or hidden_size // num_attention_heads)

    rope_params = getattr(tc, "rope_parameters", None) or {}
    # Qwen3.5 uses partial RoPE; __post_init__ defaults partial_rotary_factor to
    # 0.25, so the rotary span is a quarter of head_dim unless overridden.
    partial_rotary_factor = float(rope_params.get("partial_rotary_factor", 0.25))
    rotary_dim = int(head_dim * partial_rotary_factor)

    linear_key_head_dim = int(tc.linear_key_head_dim)
    linear_value_head_dim = int(tc.linear_value_head_dim)
    linear_num_key_heads = int(tc.linear_num_key_heads)
    linear_num_value_heads = int(tc.linear_num_value_heads)
    key_dim = linear_key_head_dim * linear_num_key_heads
    value_dim = linear_value_head_dim * linear_num_value_heads
    conv_dim = key_dim * 2 + value_dim

    layer_types = getattr(tc, "layer_types", None)
    if layer_types is None:
        layer_types = []
    layer_types = tuple(layer_types)

    return Qwen35KernelDims(
        hidden_size=hidden_size,
        intermediate_size=int(tc.intermediate_size),
        num_hidden_layers=int(tc.num_hidden_layers),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(tc.num_key_value_heads),
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rms_norm_eps=float(tc.rms_norm_eps),
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        linear_conv_kernel_dim=int(tc.linear_conv_kernel_dim),
        conv_dim=conv_dim,
        layer_types=layer_types,
    )
