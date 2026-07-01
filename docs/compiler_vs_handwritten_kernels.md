# Full `torch.compile` Artifact vs Handwritten Kernels

This document compares the handwritten Qwen3.5 Triton path against the actual
code shape produced by `torch.compile` for the full fixed-shape decode step.

This is not a toy leaf-op comparison. The artifact below comes from compiling a
real Qwen3.5 one-token decode forward with a reused `StaticCache`.

## Setup

Configuration used for the artifact audit:

- Model: random-init Qwen3.5 0.8B config, same code path as the benchmark sweep
- Timed/compiled region shape: one-token decode after a 32-token prefill
- Cache: Hugging Face `StaticCache`
- Compiler call: `torch.compile(model, mode="max-autotune", fullgraph=False)`
- Device: H100
- PyTorch: `2.12.1+cu130`
- Debug mode: `TORCH_COMPILE_DEBUG=1`

The generated raw file was an Inductor `output_code.py` under `/tmp`. It is not
committed because these files are version-, shape-, and device-specific.

Observed artifact shape:

| Item | Observed |
|---|---:|
| Inductor `output_code.py` partitions | 1 |
| Generated Triton function definitions | 37 |
| Triton launch call sites in wrapper | 267 |

This already answers the most important question: the full compiled decode step
does not map one handwritten kernel to one generated kernel. Inductor lowers the
whole fixed-shape decode graph into a large wrapper with many generated Triton
kernels, repeated launches across layers, and calls/regions corresponding to
attention, DeltaNet fast-path ops, cache updates, matmuls, normalizations, and
pointwise math.

## Representative Generated Kernels

The examples below are excerpts from the full decode-step artifact, not from
manually segmented standalone functions.

### MLP: compiler fuses across the handwritten boundary

Representative generated kernel:

```text
triton_red_fused__unsafe_view_mm_mul_silu_t_view_10
```

Inductor provenance comment:

```text
Topologically Sorted Source Nodes:
[linear_5, silu, linear_6, mul_7, down_proj]

Original ATen:
[aten.mm, aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t]
```

Excerpt from the generated Triton body:

```python
tmp0 = tl.load(in_ptr0 + r0_1, ...)
tmp9 = tl.load(in_ptr1 + r0_1, ...)
tmp13 = tl.load(in_ptr2 + (r0_1 + 3584*x0), ...).to(tl.float32)

tmp3 = -tmp0.to(tl.float32)
tmp4 = libdevice.exp(tmp3)
tmp7 = tmp0.to(tl.float32) / (tmp4 + 1.0)  # silu(gate)
tmp11 = tmp7 * tmp9.to(tl.float32)         # silu(gate) * up
tmp15 = tmp11 * tmp13.to(tl.float32)       # fused into matmul reduction
_tmp17 = tl.where(r0_mask & xmask, _tmp17 + tmp15, _tmp17)
...
tl.store(out_ptr0 + x0, tl.sum(_tmp17, 1)[:, None], xmask)
```

The handwritten path in this repo patches only the elementwise middle:

```python
return self.down_proj(fused_silu_mul(self.gate_proj(x), self.up_proj(x)))
```

So the manual kernel fuses:

```text
silu(gate_proj(x)) * up_proj(x)
```

but the compiled artifact can fuse the activation/multiply into the following
`down_proj` reduction for this decode shape. That is a larger fusion boundary
than the handwritten leaf kernel. This is one concrete reason the compiler
baseline should not be treated as "eager plus a few obvious fusions"; it can see
through module boundaries that the monkey patch intentionally preserves.

### Q/K norm + RoPE: compiler emits a large fused pointwise region

Representative generated kernel:

```text
triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mm_mul_neg_pow_rsqrt_select_sin_slice_transpose_unsqueeze_view_26
```

Inductor provenance comment, shortened:

```text
Topologically Sorted Source Nodes:
[position id construction, cos, sin, linear_25, view_3, mean_8,
 add_28, rsqrt_8, output_19, key_states, k_rot, neg, cat, k_embed]

Original ATen:
[aten.unsqueeze, aten._to_copy, aten.expand, aten.view, aten.arange,
 aten.add, aten.slice, aten.bmm, aten.transpose, aten.select, aten.cat,
 aten.cos, aten.mul, aten.sin, aten.mm, aten.pow, aten.mean, aten.rsqrt,
 aten.neg]
```

Excerpt from the generated Triton body:

```python
tmp46 = tl.load(in_ptr0 + (x0 + 256*x1), xmask)
tmp49 = tl.load(in_ptr1 + x1, xmask)
tmp56 = tl.load(in_ptr2 + x0, xmask).to(tl.float32)
...
tmp12 = (tmp8 / 256.0) + 1e-06
tmp13 = libdevice.rsqrt(tmp12)
tmp14 = tmp7 * tmp13
tmp19 = tmp14 * (tmp15.to(tl.float32) + 1.0)
tmp21 = -tmp19
...
```

The handwritten Qwen3.5 package contains an optional `fused_qknorm_rope` kernel,
but the default benchmark patch path does not enable the whole-attention
replacement. It only swaps `Qwen3_5RMSNorm.forward`, which also affects Q/K norm
modules. The compiled full artifact, by contrast, sees the surrounding position
construction, norm, trig tables, slicing, rotation, and embedding/cache logic as
one larger fused region where possible.

This is not a claim that the generated Q/K-RoPE region is always superior to a
carefully handwritten kernel. It is evidence that the fair compiler baseline is
not merely reproducing the same small hand-written leaf kernel; it is compiling
a broader graph region.

### Attention mask / SDPA boundary

Representative generated kernel:

```text
triton_poi_fused__scaled_dot_product_cudnn_attention__unsafe_view_add_arange_clone_expand_le_scalar_tensor_unsqueeze_where_1
```

Inductor provenance comment:

```text
Topologically Sorted Source Nodes:
[key_6, value_6, arange_4, kv_arange, kv_indices, arange_3, q_arange,
 q_indices, attention_mask, attention_mask_1, attn_output]

Original ATen:
[aten.unsqueeze, aten.expand, aten.clone, aten._unsafe_view, aten.arange,
 aten.add, aten.le, aten.scalar_tensor, aten.where,
 aten._scaled_dot_product_cudnn_attention]
```

Excerpt:

```python
tmp0 = tl.load(in_ptr0 + 0)
tmp3 = tmp0 + 0
tmp5 = x0 <= tmp3
tmp8 = tl.where(tmp5, 0.0, float("-inf"))
tl.store(out_ptr0 + x0, tmp8, xmask)
```

The custom Triton path does not replace attention with a handwritten attention
kernel. The compiled graph still relies on the framework attention backend, but
it also folds surrounding mask/index construction into generated kernels. That
matters for decode workloads because CPU launch overhead and small pointwise
ops are part of the cost.

### DeltaNet / causal-conv fast path stays part of the compiled graph

Representative generated kernel:

```text
triton_per_fused__causal_conv1d_update_cpp__to_copy__unsafe_view_add_embedding_mean_mm_mul_pow_rsqrt_squeeze_t_transpose_view_5
```

Inductor provenance comment:

```text
Topologically Sorted Source Nodes:
[inputs_embeds, RMSNorm pieces, hidden_states, mixed_qkv, squeeze]

Original ATen:
[aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt,
 aten.mul, aten.view, aten.mm, aten.t, aten._unsafe_view, aten.transpose,
 aten.squeeze, DaoAILab._causal_conv1d_update_cpp]
```

The important detail is that the fair Qwen3.5 baseline must have the
`flash-linear-attention` / `causal-conv1d` fast path installed. The handwritten
Triton kernels are not replacing that subsystem. The compiled artifact shows
those fast-path ops participating in the same compiled decode graph.

## Comparison Against The Handwritten Path

The handwritten path patches selected module boundaries:

| Handwritten patch | Scope |
|---|---|
| `rmsnorm` | Qwen3.5 RMSNorm, including the `(1 + weight)` convention |
| `fused_silu_mul` | only the `silu(gate) * up` elementwise middle of the MLP |
| `rmsnorm_gated` | DeltaNet output norm plus gate |
| optional `fused_qknorm_rope` | Q/K norm + partial RoPE, only when the whole attention-forward patch is enabled |
| `GraphedDecoder` | manual CUDA graph replay over a fixed-shape decode step |

The compiled path is different. It compiles the full one-token decode forward
with static cache state and emits many generated kernels across larger graph
regions. In the observed artifact, some generated kernels include `mm`, `silu`,
normalization, RoPE-related pointwise work, attention-mask construction, cache
updates, and DeltaNet fast-path boundaries in the same compiled wrapper.

That means the comparison is not:

```text
handwritten RMSNorm kernel vs compiler RMSNorm kernel
```

The real comparison is:

```text
handwritten leaf patches + manual graph replay
vs
compiler-generated full decode-step wrapper + generated Triton regions + framework fast paths + graph/replay behavior
```

## Takeaway

The full artifact makes the README result more plausible, not less:

- The handwritten kernels optimize useful local regions.
- The compiler baseline can generate Triton for broader regions than those
  handwritten leaf patches.
- For MLP, the generated artifact fuses `silu(gate) * up` into a following
  matmul-style reduction, while the handwritten patch leaves `down_proj` as a
  separate call.
- For Q/K norm and RoPE-related work, the generated artifact includes a large
  pointwise region around position construction, normalization, trig tables, and
  rotation/cache work.
- The largest benchmark gains still come from static cache and graph/compile
  replay, not from the existence of a few handwritten kernels.

This is why a strong baseline must be `StaticCache + torch.compile` or an
equivalent fixed-shape graph path. Eager `generate()` is not a serious
denominator for attributing decode speedups to custom kernels.

## Caveats

- This artifact is for Qwen3.5 0.8B, batch 1, one-token decode, H100,
  PyTorch `2.12.1+cu130`. Other sizes and PyTorch versions can generate
  different code.
- The model is random-initialized, as in the latency benchmark. That is valid
  for compile/code-shape and latency attribution, not for quality evaluation.
- This document inspects Qwen3.5 full-decode compile output. Qwen3-TTS has a
  separate benchmark and a custom compile wrapper over predictor/talker regions;
  its full Inductor artifact is not included here.
