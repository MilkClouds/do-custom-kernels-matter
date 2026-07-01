# Compiler-Generated vs Handwritten Kernels

This note compares what the handwritten kernels in this repo do against what a
proper `torch.compile` baseline can generate. It is intentionally separate from
the README: generated compiler artifacts are useful for audit, but they are too
version-, shape-, and device-specific to be the main public narrative.

## What `torch.compile` Produces

`torch.compile` does not emit one readable replacement for a whole model. The
pipeline captures an FX graph, lowers pieces through Inductor, and emits a
Python wrapper that launches a mix of:

- generated Triton kernels for fused pointwise/reduction regions,
- library calls such as cuBLAS/cuDNN/attention backends for heavy kernels,
- graph replay / reduced dispatch overhead when the graph is static enough.

So a literal line-by-line comparison is only meaningful for small isolated
regions. For full autoregressive decode, the better comparison is structural:
which operations are fused, which shapes are static, whether the decode step is
capturable, and what the end-to-end attribution says.

## Isolated Op Check

To sanity-check the compiler side, three small functions matching the
manual-kernel targets were compiled on an H100 with PyTorch `2.12.1+cu130`,
bf16 tensors, and `mode="max-autotune"`. Dumps were generated only in `/tmp` with
`TORCH_COMPILE_DEBUG=1`; raw generated files are not committed.

| Function compiled | Inductor output | Interpretation |
|---|---|---|
| RMSNorm over `[16, 4096]` | one Triton reduction kernel, `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | Inductor fuses cast, square, mean, rsqrt, multiply, and output cast into one row-wise kernel. |
| `silu(gate) * up` over `[16, 12288]` | one Triton pointwise kernel, `triton_poi_fused__to_copy_mul_silu_0` | Inductor emits the same basic fusion class as the handwritten SwiGLU/SILU-mul kernel. |
| residual add + RMSNorm over `[16, 4096]` | one Triton reduction kernel, `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | Inductor can fuse the residual add into the RMSNorm-style reduction when the region is compiled as one graph. |

This does not prove that every full-model compile will fuse every instance the
same way. It does show that the manual kernels are not uniquely capable of these
leaf fusions: Inductor can generate the same categories of Triton kernels when
the code is presented as a stable, compilable region.

## Handwritten Kernel Scope

### Qwen3.5

The vendored Qwen3.5 path patches selected leaf ops:

| Manual kernel | File | What it replaces |
|---|---|---|
| RMSNorm | `src/qwen35_triton/kernels/rmsnorm.py` | Qwen3.5 RMSNorm, including the `(1 + weight)` convention. |
| SwiGLU / SiLU-mul | `src/qwen35_triton/kernels/mlp.py` | The elementwise `silu(gate_proj(x)) * up_proj(x)` between cuBLAS GEMMs. |
| gated RMSNorm | `src/qwen35_triton/kernels/rmsnorm_gated.py` | DeltaNet output norm plus gate. |
| QK-norm + partial RoPE | `src/qwen35_triton/kernels/qknorm_rope.py` | Optional fusion inside attention; the default patch path already covers QK-norm through the RMSNorm swap. |

The GEMMs remain cuBLAS. Attention and DeltaNet fast paths remain external
framework/library work. The custom kernels therefore optimize small reduction
and pointwise islands; they do not replace the full decode step.

### Qwen3-TTS

The public `qwen3-tts-triton` Hybrid path benchmarked here patches:

| Manual kernel family | Public file | What it replaces |
|---|---|---|
| RMSNorm | `qwen3_tts_triton/kernels/rms_norm.py` | RMSNorm module calls. |
| SwiGLU | `qwen3_tts_triton/kernels/swiglu.py` | `silu(gate) * up` inside MLP. |
| residual add + RMSNorm | `qwen3_tts_triton/kernels/fused_norm_residual.py` | Post-attention residual add plus norm. |

That repository also contains an M-RoPE kernel, but the audited
`apply_triton_kernels()` path patches RMSNorm, MLP SwiGLU, and fused
norm/residual. The large Hybrid speedup comes from combining these patches with
the `faster-qwen3-tts` StaticCache + CUDA graph path, not from the leaf kernels
alone.

## What the Comparison Says

The compiler and handwritten kernels are closer at the leaf-op level than the
marketing framing suggests. For RMSNorm, SiLU-mul, and residual+RMSNorm,
Inductor can produce one fused Triton kernel for the same broad computation
class. Handwritten Triton can still be useful for exact numerical control,
special layouts, version-independent patching, or cases where the compiler
misses a fusion. But those are local benefits.

The end-to-end results are dominated by a larger lever:

| Case | Leaf-kernel effect | Static/graph effect | Strong framework result |
|---|---:|---:|---|
| Qwen3.5 9B decode | 1.08-1.15x | 2.38-3.32x | `StaticCache + torch.compile(max-autotune)` beats custom+graph by 1.06-1.49x across 0.8B-27B. |
| Qwen3-TTS E2E | 1.29-1.35x over Faster | 3.75-5.74x over Base | fixed-shape predictor/talker `torch.compile(max-autotune)` is 1.46-1.64x faster than Hybrid. |

That is the main attribution point. A handwritten Triton kernel may be locally
reasonable and still be the wrong explanation for a headline 4-10x improvement.
For decode-style workloads, first make the KV/state cache static, compile or
graph the fixed-shape step, and only then measure whether custom kernels add
meaningful marginal speed.

## How To Audit This Properly

Use compiler artifacts as supporting evidence, not as the primary benchmark.
Generated code changes across PyTorch versions and shapes, and full-model
decode includes library calls plus graph replay, not just generated Triton.

The robust audit sequence is:

1. Check correctness for the custom kernels.
2. Measure eager/static-cache, graph-only, kernel-only, kernel+graph, and
   `torch.compile` paths on the same timed region.
3. Inspect generated compiler output for representative leaf ops to see whether
   the compiler already fuses the same operation class.
4. Attribute the end-to-end speedup to the lever that actually moves the number.
