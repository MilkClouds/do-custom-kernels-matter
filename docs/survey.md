# Survey Summary

Static audit over public repos advertising GPU-kernel inference optimization.
The question is attribution: kernel work vs graph/cache work.

## New Empirical Check

For [`newgrit1004/qwen3-tts-triton`](https://github.com/newgrit1004/qwen3-tts-triton),
the repo's own ablation says the custom Triton layer is a marginal improvement
on top of the graph/static-cache path. Our A100/H100 E2E runs strengthen that:

H100:

| Path | Mean latency | ms / 1k samples |
|---|---:|---:|
| Faster | 1429 ms | 10.63 |
| Hybrid | 1093 ms | 7.90 |
| `torch.compile(reduce-overhead)` | 838 ms | 6.06 |
| `torch.compile(max-autotune)` | 551 ms | **4.78** |
| Hybrid + `compile(reduce-overhead)` | 747 ms | 6.08 |
| Hybrid + `compile(max-autotune)` | 683 ms | 5.08 |

A100:

| Path | Mean latency | ms / 1k samples |
|---|---:|---:|
| Faster | 1949 ms | 13.53 |
| Hybrid | 1479 ms | 10.41 |
| `torch.compile(reduce-overhead)` | 1133 ms | 7.66 |
| `torch.compile(max-autotune)` | 837 ms | **6.51** |
| Hybrid + `compile(reduce-overhead)` | 981 ms | 7.99 |
| Hybrid + `compile(max-autotune)` | 1117 ms | 6.76 |

The compile baseline targets the same two regions as `faster_qwen3_tts`:
`PredictorGraph._full_loop` and `TalkerGraph._decode_step`. It does not compile
outer `generate()`. The compile and Hybrid + compile rows use `attn=sdpa`;
Base/Faster/Hybrid use the public repo defaults. `max-autotune` rows are
separate compile-only runs. Hybrid + compile applies the repo's Triton patches
inside the compiled regions; it lands at 0.94-1.00x of the plain compile path
per generated sample, so the custom kernels add nothing once the regions are
compiled.

## Scope

This file is a public-examples summary, not a full raw audit log. Treat it as
secondary context for the reproduced Qwen3.5 and Qwen3-TTS measurements in the
README.

## Public materials worth reading carefully

Star counts were observed on 2026-07-01 and are included only as visibility
context.

| Repo | Stars | Attribution note |
|---|---:|---|
| [`harleyszhang/lite_llama`](https://github.com/harleyszhang/lite_llama) | 188 | Optimized path includes compiled/static execution, so eager HF generate + DynamicCache is not enough to isolate kernel contribution. |
| [`tsdocode/nano-qwen3tts-vllm`](https://github.com/tsdocode/nano-qwen3tts-vllm) | 130 | README itself credits CUDA Graphs for the 2-3x path. |
| [`RightNow-AI/qwen3.5-triton`](https://github.com/RightNow-AI/qwen3.5-triton) | 117 | Own docs separate kernels-in-eager (~1.4x) from CUDA graph (~2.2x); no steel-man static-cache compile baseline. |
| [`newgrit1004/qwen3-tts-triton`](https://github.com/newgrit1004/qwen3-tts-triton) | 96 | Transparent split: Triton-only ~1.1x; Faster graph/static-cache ~4x; Hybrid ~5x. |
| [`newgrit1004/omnivoice-triton`](https://github.com/newgrit1004/omnivoice-triton) | 54 | Own table: Triton-only 1.02x; CUDA-Graph-only 2.75x. |
| [`RightNow-AI/TIDE`](https://github.com/RightNow-AI/TIDE) | 32 | Reported speedups require path-level benchmark audit before attributing gains to kernels alone. |
| [`Vishwa44/SparkTTSOptimized`](https://github.com/Vishwa44/SparkTTSOptimized) | 0 | Own table: `torch.compile + FA2` beats custom kernels. |
| [`Hmbown/reCUDA`](https://github.com/Hmbown/reCUDA) | 0 | Precision regime and full-model-vs-microbenchmark attribution need to be read separately; reported full-model gain is ~1.1-1.2x. |

## Pattern

The repeated pattern is simple:

1. Baseline is eager decode, often with DynamicCache.
2. Optimized path uses static shapes, cache reuse, CUDA graphs, or vLLM-style graph capture.
3. The headline says "Triton/custom kernel" even when the kernel-only delta is near 1.0-1.3x.

Structured public examples: [`results/genre_survey.json`](../results/genre_survey.json).
