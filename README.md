# Do Custom Kernels Matter?

LLMs can now write plausible Triton/CUDA kernels, and some projects are getting
attention by using them for inference optimization. The important question is
not whether the kernels look impressive, but whether they actually explain the
end-to-end speedup.

This repo asks one narrow question:

> When an inference repo adds custom kernels, do those kernels still matter
> after comparing against strong framework baselines such as static KV cache,
> CUDA graphs, batching, and `torch.compile`?

Custom kernels and `torch.compile` are not mutually exclusive, since a Triton
kernel can run inside a compiled region. The question is therefore measured
two ways: custom-kernel path vs `torch.compile` path head-to-head, and custom
kernels **added on top of** the same `torch.compile` path.

Short answer: not by themselves. In the reproduced cases below, the large
headline speedups come from static shapes and graph capture. The handwritten
kernels add a much smaller marginal improvement, a **correctly configured
`torch.compile` baseline can match or beat the custom-kernel path**, and
stacking the custom kernels on top of that compile baseline does not improve
it. This is a claim about end-to-end decode/generation attribution, not a
claim that custom kernels are never useful.

This is not an anti-kernel argument. [FlashAttention](https://github.com/Dao-AILab/flash-attention),
[Liger Kernel](https://github.com/linkedin/Liger-Kernel),
[Hugging Face kernels](https://github.com/huggingface/kernels), and tiled DSLs
such as [TileLang](https://github.com/tile-ai/tilelang) are real parts of the
kernel optimization ecosystem. The point is measurement: compare custom kernels
against the strongest framework baseline, not only against eager `generate()`.

## Main Results

All numbers are steady-state measurements after warmup. Qwen3.5
framework-vs-custom is the 0.8B-27B sweep; its attribution columns are the 9B
ablation. Qwen3-TTS ratios use output-normalized latency (`ms / 1k samples`).

| Case | Custom-kernel path | Best framework baseline | Framework faster than custom | Kernel-only gain | Static/graph gain | Kernel-on-compile gain | Takeaway |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.5 decode | [`RightNow-AI/qwen3.5-triton`](https://github.com/RightNow-AI/qwen3.5-triton)-derived kernels + manual graph | `StaticCache` + `torch.compile(max-autotune)` | **1.11-1.50x** | 1.07-1.11x | **2.39-2.97x** | 0.99-1.02x | compile wins; graph dominates; kernels redundant under compile |
| Qwen3-TTS E2E | [`newgrit1004/qwen3-tts-triton`](https://github.com/newgrit1004/qwen3-tts-triton) Hybrid | fixed-shape predictor/talker `torch.compile(max-autotune)` | **1.60-1.65x** | 1.31-1.32x | **3.55-4.80x** | 0.94-1.00x | compile wins; graph dominates; kernels redundant under compile |

Read the table as:

- `Framework baseline`: PyTorch/Hugging Face/compiler/runtime optimizations,
  without project-specific handwritten Triton/CUDA kernels.
- `Framework faster than custom`: `custom latency / framework latency` for
  Qwen3-TTS, and `framework tok/s / custom tok/s` for Qwen3.5. Values above
  **1.0x mean the framework baseline is faster**.
- `Kernel-only gain`: custom kernels over the matching framework path.
- `Static/graph gain`: graph/static-cache path over the ordinary baseline.
- `Kernel-on-compile gain`: the same custom kernels applied *inside* the
  `torch.compile` path, versus that compile path alone. Values around or below
  1.0x mean the kernels add nothing once the compiler is in play.

Survey snapshot: the same attribution pattern appears in other public repos,
but those examples are secondary context. The reproduced results above are the
primary evidence.

## Measurement Model

A custom kernel is GPU code that replaces one or more framework operations. It
is often written in [Triton](https://github.com/triton-lang/triton), CUDA, or a
tiled DSL such as [TileLang](https://github.com/tile-ai/tilelang). It can help
by reducing memory traffic, fusing operations, improving locality, changing the
algorithm, or removing launches. It does not automatically help when runtime is
dominated by cuBLAS GEMMs, Python overhead, unstable cache shapes, or graph
capture.

Autoregressive inference has two different phases:

| Phase | Workload | Common bottleneck | Typical optimization |
|---|---|---|---|
| Prefill | Process many prompt tokens at once | compute and memory bandwidth | FlashAttention, batching, better attention backend |
| Decode | Generate one token/frame at a time | launch overhead, cache shape, small ops | StaticCache, CUDA graphs, `torch.compile`, batching |

Key terms:

| Term | Why it matters |
|---|---|
| KV cache | Stores attention keys/values from previous tokens so decode does not recompute the whole prefix. |
| DynamicCache | Grows as generation proceeds. Convenient, but changing shapes and allocations make it a weak optimized baseline. |
| StaticCache | Preallocates a fixed-size KV cache. Shapes and memory addresses stay stable, which is what CUDA graphs and `torch.compile(reduce-overhead)` need for fast replay. |
| CUDA graph | Captures a fixed GPU execution pattern and replays it with much lower CPU/dispatcher overhead. |
| `torch.compile` | PyTorch's compiler path. In this repo, `reduce-overhead` targets launch overhead and graph replay; `max-autotune` spends more compile time searching faster optimized kernels. |

The roofline lens is useful: first ask whether the workload is compute-bound,
memory-bound, or launch-bound.

![Roofline performance model from NERSC Documentation](https://docs.nersc.gov/tools/performance/roofline/Roofline-intro.png)

Figure from the
[NERSC Roofline Performance Model documentation](https://docs.nersc.gov/tools/performance/roofline/).
It is a conceptual performance envelope, not measured data from this repo.

| Regime | Main limit | What tends to help |
|---|---|---|
| Compute-bound | arithmetic throughput | better algorithms, better GEMM shapes, tensor cores |
| Memory-bound | reads/writes and bandwidth | fusion, fewer intermediate tensors, better locality |
| Launch-bound | many tiny kernels and dispatcher overhead | CUDA graphs, `torch.compile(reduce-overhead)`, batching |

A fair report should separate cache-shape, graph/launch, and custom-kernel
effects. The useful denominator is the best framework static-cache/compile path,
not eager DynamicCache.

One objection to a "custom kernels vs `torch.compile`" table is that the two
are not exclusive: `torch.compile` traces user-defined Triton kernels, so the
custom kernels can be applied *inside* the compiled decode step. Both views are
measured here. The head-to-head comparison answers "do the kernels beat the
strongest zero-effort baseline?"; the stacked path (`Custom + compile`,
`Hybrid + compile`) answers "do the kernels add anything the compiler was not
already doing?". For launch-bound decode the custom kernels and Inductor's own
codegen compete for the same fusion opportunities, so the stacked path is the
direct test of whether the handwritten kernels are complementary or redundant.

For more background, see the PyTorch
[`torch.compile` docs](https://docs.pytorch.org/docs/2.12/generated/torch.compile.html),
the Hugging Face
[Accelerate compilation guide](https://huggingface.co/docs/accelerate/usage_guides/compilation),
and the PyTorch [GPT-Fast blog](https://pytorch.org/blog/accelerating-generative-ai-2/),
which uses static KV cache and `torch.compile(reduce-overhead)` for LLM decode.

## Case Study 1: Qwen3.5 Decode

This tests the public
[`RightNow-AI/qwen3.5-triton`](https://github.com/RightNow-AI/qwen3.5-triton)
kernel idea in isolation. Qwen3.5 models are random-initialized Hugging Face
configs, which is enough for latency and attribution. The Qwen3.5 fast path
dependencies
[`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
and [`causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d) are required.
The vendored code under `src/qwen35_triton` is derived from that public
MIT-licensed project; upstream attribution is in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The public implementation is
oriented around Qwen3.5-27B, while this repo generalizes dimensions from the
Hugging Face config so the same custom-kernel path can be tested on 0.8B, 2B,
4B, 9B, and 27B.

Benchmark setup:

- Fixed 512-token prompt, 128 generated tokens.
- Every path uses a reused `StaticCache`.
- `torch.compile` is applied to the fixed-shape one-token decode step with
  stable cache buffers and `cache_position`.
- A DynamicCache or per-call cache allocation would make the framework baseline
  artificially weak.

Paths:

| Path | Meaning |
|---|---|
| Eager | HF decode with reused `StaticCache` |
| `compile(reduce-overhead)` | HF static decode compiled for lower launch overhead |
| `compile(max-autotune)` | stronger framework compiler baseline |
| Custom + graph | generalized RightNow-AI-derived Triton patches + manual CUDA graph |
| Custom + compile | the same Triton patches applied inside the `torch.compile` decode step |

H100, bf16, batch 1, fixed 512-token prompt, 128 generated tokens:

| Model | Eager | `compile(reduce-overhead)` | `compile(max-autotune)` | Custom + graph | Custom + `compile(reduce-overhead)` | Custom + `compile(max-autotune)` |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B | 51.9 tok/s | 561.1 | 638.0 | 424.3 | 482.8 | 644.6 |
| 2B | 53.3 tok/s | 380.3 | 452.0 | 331.1 | 369.8 | 457.5 |
| 4B | 37.4 tok/s | 206.8 | 238.7 | 186.1 | 204.7 | 239.6 |
| 9B | 37.1 tok/s | 140.2 | 156.3 | 130.6 | 138.9 | 160.3 |
| 27B | 18.5 tok/s | 46.9 | 50.4 | 44.3 | 46.6 | 49.4 |

A100, same configuration:

| Model | Eager | `compile(reduce-overhead)` | `compile(max-autotune)` | Custom + graph | Custom + `compile(reduce-overhead)` | Custom + `compile(max-autotune)` |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B | 40.6 tok/s | 367.9 | 428.3 | 315.5 | 360.1 | 396.7 |
| 2B | 39.4 tok/s | 250.7 | 288.5 | 224.8 | 245.4 | 278.8 |
| 4B | 30.5 tok/s | 129.0 | 140.9 | 119.0 | 127.0 | 133.6 |
| 9B | 30.0 tok/s | 84.2 | 89.2 | 79.2 | 82.1 | 89.8 |
| 27B | 14.8 tok/s | 26.4 | 28.2 | 25.5 | 26.1 | 27.7 |

Adding the custom kernels inside the compiled decode step moves it by
0.86-1.03x across the sweep: around 1.0x at every size, slightly negative
more often than positive, and the only large deviation (0.86x, 0.8B
`reduce-overhead` on H100) is a slowdown. The patched decode step compiles
with zero graph breaks (`TORCH_LOGS=graph_breaks`), so the Triton kernels do
run inside the compiled region; they are simply not better than the fused
norm/MLP elementwise kernels Inductor generates on its own.

9B attribution ablation:

| Device | Eager | HF graph | Custom, no graph | Custom + graph | Kernel-only | Graph-only | Kernel on graph |
|---|---:|---:|---:|---:|---:|---:|---:|
| H100 | 37.0 | 109.9 | 41.2 | 130.0 | 1.11x | 2.97x | 1.18x |
| A100 | 29.7 | 71.1 | 31.9 | 80.3 | 1.07x | 2.39x | 1.13x |

9B compile stacking, same runs:

| Device | `compile(reduce-overhead)` | Custom + compile(RO) | `compile(max-autotune)` | Custom + compile(MA) | Kernel on compile |
|---|---:|---:|---:|---:|---:|
| H100 | 140.3 | 138.9 | 156.9 | 159.8 | 0.99-1.02x |
| A100 | 83.9 | 83.4 | 91.7 | 92.6 | 0.99-1.01x |

Artifacts:
[`qwen35_decode_h100.json`](results/qwen35_decode_h100.json),
[`qwen35_decode_a100.json`](results/qwen35_decode_a100.json),
[`qwen35_ablation_h100.json`](results/qwen35_ablation_h100.json),
[`qwen35_ablation_a100.json`](results/qwen35_ablation_a100.json).

## Case Study 2: Qwen3-TTS E2E

`qwen3-tts-triton` uses these labels:

| Label | Meaning |
|---|---|
| Base | ordinary PyTorch/Qwen TTS runner |
| Faster | `faster-qwen3-tts`: StaticCache + CUDA graph, no repo Triton patches |
| Hybrid | Faster + repo custom Triton patches |
| Hybrid + compile | the same repo Triton patches applied inside the compile rows' regions |

Benchmark setup:

- Timed region: steady-state end-to-end `generate_custom_voice` through audio
  output.
- Excluded from timing: model loading, graph capture, and compilation.
- Compile target: the same fixed-shape regions that `faster-qwen3-tts` graphs,
  namely the 15-codebook predictor loop and the one-step talker decode.
- Not compiled: the outer Python `generate()` wrapper.
- Compile implementation: stable static cache, masks, and input buffers in
  [`src/kernelcheck/compiled_qwen3_tts.py`](src/kernelcheck/compiled_qwen3_tts.py).
- Hybrid + compile applies the repo's Triton patches with the same settings as
  Hybrid (`enable_fused_norm=True`, layers 0-23) to the compile runner's model
  before compile warmup, so the compiled regions trace the custom kernels.
- Attention: Base/Faster/Hybrid use public repo defaults; compile and
  Hybrid + compile rows use `attn=sdpa`.
- Metric: output-normalized `ms / 1k samples`, because output length differs
  across paths (greedy decode diverges once numerics differ; the gap reaches
  ~30% between compile and Hybrid + compile rows). Faster and Hybrid differ by
  1-3%, so the kernel-only comparison is more direct.

H100, end-to-end `generate_custom_voice`, `max_new_tokens=256`:

| Path | Mean latency | Output samples | ms / 1k samples |
|---|---:|---:|---:|
| Base | 4782 ms | 126,720 | 37.74 |
| Faster | 1429 ms | 134,400 | 10.63 |
| `compile(reduce-overhead)` | 838 ms | 138,240 | 6.06 |
| `compile(max-autotune)` | 551 ms | 115,200 | **4.78** |
| Hybrid | 1093 ms | 138,240 | 7.90 |
| Hybrid + `compile(reduce-overhead)` | 747 ms | 122,880 | 6.08 |
| Hybrid + `compile(max-autotune)` | 683 ms | 134,400 | 5.08 |

A100, same configuration:

| Path | Mean latency | Output samples | ms / 1k samples |
|---|---:|---:|---:|
| Base | 6733 ms | 103,680 | 64.94 |
| Faster | 1949 ms | 144,000 | 13.53 |
| `compile(reduce-overhead)` | 1133 ms | 147,840 | 7.66 |
| `compile(max-autotune)` | 837 ms | 128,640 | **6.51** |
| Hybrid | 1479 ms | 142,080 | 10.41 |
| Hybrid + `compile(reduce-overhead)` | 981 ms | 122,880 | 7.99 |
| Hybrid + `compile(max-autotune)` | 1117 ms | 165,120 | 6.76 |

The stacked rows repeat the Qwen3.5 result: adding the repo's Triton patches
inside the compiled regions gives 0.94-1.00x per generated sample, and the
compile path alone stays the fastest on both devices.

Artifacts:
[`qwen3_tts_h100_reduce.json`](results/qwen3_tts_h100_reduce.json),
[`qwen3_tts_h100_max_autotune.json`](results/qwen3_tts_h100_max_autotune.json),
[`qwen3_tts_a100_reduce.json`](results/qwen3_tts_a100_reduce.json),
[`qwen3_tts_a100_max_autotune.json`](results/qwen3_tts_a100_max_autotune.json).

## Other Public Examples

These are public materials worth reading with the same attribution lens. They
are examples where the reported speedup should not be read as "the custom
kernel did it" without checking the baseline, cache shape, graph capture,
batching, and precision regime. Star counts were observed on 2026-07-01.

| Repo | Stars | Why it is relevant |
|---|---:|---|
| [`harleyszhang/lite_llama`](https://github.com/harleyszhang/lite_llama) | 188 | Optimized path includes compiled/static execution, so eager HF `generate()` is not enough to isolate kernel contribution. |
| [`tsdocode/nano-qwen3tts-vllm`](https://github.com/tsdocode/nano-qwen3tts-vllm) | 130 | README credits CUDA graphs for the main 2-3x path. |
| [`newgrit1004/omnivoice-triton`](https://github.com/newgrit1004/omnivoice-triton) | 54 | Own mode table separates Triton-only 1.02x from CUDA-Graph-only 2.75x. |
| [`RightNow-AI/TIDE`](https://github.com/RightNow-AI/TIDE) | 32 | Reported speedups require path-level benchmark audit before attributing gains to kernels alone. |
| [`Vishwa44/SparkTTSOptimized`](https://github.com/Vishwa44/SparkTTSOptimized) | 0 | Own benchmark table includes a `torch.compile + FA2` path that beats the custom-kernel path. |
| [`Hmbown/reCUDA`](https://github.com/Hmbown/reCUDA) | 0 | Precision regime and full-model-vs-microbenchmark attribution need to be read separately; reported full-model gain is around 1.1-1.2x. |

Survey summary: [`docs/survey.md`](docs/survey.md). Structured summary:
[`results/genre_survey.json`](results/genre_survey.json).

## Reproduce

Install:

```bash
uv sync --extra qwen35 --group dev
```

The `qwen35` extra currently assumes Linux x86_64, CPython 3.11, CUDA 13, and
the pinned `causal-conv1d` wheel in [`pyproject.toml`](pyproject.toml).

Qwen3.5 kernel correctness gate:

```bash
uv run --extra qwen35 python -u scripts/qwen35/check_correctness.py --device cuda
```

Qwen3.5 decode:

```bash
uv run --extra qwen35 python -u scripts/qwen35/bench_decode.py \
  --device cuda \
  --sizes 0.8B 2B 4B 9B 27B \
  --paths eager compile patched-graph patched-compile \
  --compile-modes reduce-overhead max-autotune \
  --json results/qwen35_decode.json
```

Qwen3.5 attribution:

```bash
uv run --extra qwen35 python -u scripts/qwen35/bench_decode.py \
  --device cuda \
  --sizes 9B \
  --paths eager compile hf-graph patched-eager patched-graph patched-compile \
  --json results/qwen35_ablation.json
```

Qwen3-TTS E2E, after installing `qwen3-tts-triton`:

```bash
QWEN3_TTS=/path/to/qwen3-tts-triton \
PYTHONPATH="$PWD/src:$QWEN3_TTS/src:$QWEN3_TTS" \
ATTN=sdpa \
COMPILE_MODE=reduce-overhead \
OUT=results/qwen3_tts_reduce.json \
"$QWEN3_TTS/.venv/bin/python" scripts/qwen3_tts/bench_e2e.py
```

Slurm matrix:

```bash
QWEN3_TTS=/path/to/qwen3-tts-triton \
bash scripts/slurm/submit_all.sh
```

## Development

```bash
uv sync --only-group dev
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

Ruff uses `line-length = 119`; CI runs the same lint and format checks. The
repo intentionally ignores `uv.lock` because `pyproject.toml` pins the relevant
time window with `exclude-newer`.
