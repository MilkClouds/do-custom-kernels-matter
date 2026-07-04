"""Reproduce the Qwen3.5 decode-control experiment.

This script measures the attribution ladder used in the README. The timed
region includes prefill for the fixed prompt plus the requested generated
tokens; compile/graph/custom interventions target the fixed-shape decode step.

1. framework HF eager decode with a reused StaticCache;
2. framework HF StaticCache + torch.compile;
3. HF + explicit CUDA-graph replay;
4. patched custom Triton kernels without graph replay;
5. patched custom Triton kernels with graph replay;
6. patched custom Triton kernels under the same torch.compile decode step.

The model is random-init. That is intentional: the experiment measures latency
and launch/cache/kernel attribution, not language quality.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from collections.abc import Callable
from typing import Any

import torch

QWEN35_CONFIGS: dict[str, dict[str, int]] = {
    "0.8B": dict(
        hidden_size=1024,
        intermediate_size=3584,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_conv_kernel_dim=4,
    ),
    "2B": dict(
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_conv_kernel_dim=4,
    ),
    "4B": dict(
        hidden_size=2560,
        intermediate_size=9216,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_conv_kernel_dim=4,
    ),
    "9B": dict(
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_conv_kernel_dim=4,
    ),
    "27B": dict(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_conv_kernel_dim=4,
    ),
}
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
DEFAULT_PATHS = ["eager", "compile", "hf-graph", "patched-eager", "patched-graph", "patched-compile"]


def fast_path_status() -> str:
    have = []
    for mod in ("fla", "causal_conv1d"):
        try:
            __import__(mod)
            have.append(mod)
        except ImportError:
            pass
    return "+".join(have) if have else "none"


def require_qwen35_triton():
    try:
        from qwen35_triton import GraphedDecoder, patch_model, unpatch_model
    except ImportError as exc:
        raise SystemExit(
            "qwen35_triton is required for graph/custom paths. It is vendored in this repo; "
            "run through `uv run` or set PYTHONPATH=$THIS/src."
        ) from exc
    return GraphedDecoder, patch_model, unpatch_model


def build_model(size: str, dtype: torch.dtype, device: str, vocab: int):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    cfg = Qwen3_5TextConfig(vocab_size=vocab, max_position_embeddings=8192, **QWEN35_CONFIGS[size])
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(cfg).to(device, dtype=dtype).eval()


@torch.no_grad()
def timed_decode(decode_fn: Callable[[int], None], tokens: int, warmup_tokens: int) -> float:
    decode_fn(warmup_tokens)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    decode_fn(tokens)
    end.record()
    torch.cuda.synchronize()
    elapsed_s = start.elapsed_time(end) / 1000.0
    return tokens / elapsed_s


def eager_loop(model, config, input_ids):
    from transformers import StaticCache

    device = input_ids.device
    prompt_len = input_ids.shape[1]

    def run(tokens: int) -> None:
        cache = StaticCache(config=config, max_cache_len=prompt_len + tokens + 8)
        out = model(
            input_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(prompt_len, device=device),
        )
        token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache_position = torch.zeros(1, dtype=torch.long, device=device)
        for i in range(tokens):
            cache_position.fill_(prompt_len + i)
            out = model(token, past_key_values=cache, use_cache=True, cache_position=cache_position)
            token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return run


def compiled_loop(model, config, input_ids, compile_mode: str, max_cache_len: int):
    from transformers import StaticCache

    device = input_ids.device
    prompt_len = input_ids.shape[1]
    compiled_step = torch.compile(model, mode=compile_mode, fullgraph=False)
    cache = StaticCache(config=config, max_cache_len=max_cache_len)
    prefill_pos = torch.arange(prompt_len, device=device)
    cache_position = torch.zeros(1, dtype=torch.long, device=device)

    def run(tokens: int) -> None:
        cache.reset()
        out = model(input_ids, past_key_values=cache, use_cache=True, cache_position=prefill_pos)
        token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for i in range(tokens):
            cache_position.fill_(prompt_len + i)
            out = compiled_step(token, past_key_values=cache, use_cache=True, cache_position=cache_position)
            token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return run


def graph_loop(model, input_ids, max_seq_len: int):
    GraphedDecoder, _, _ = require_qwen35_triton()
    decoder = GraphedDecoder(model, max_seq_len=max_seq_len)

    def run(tokens: int) -> None:
        decoder.generate(input_ids, max_new_tokens=tokens)

    return run


def measure(label: str, fn: Callable[[], float], baseline: float | None = None) -> dict[str, Any]:
    try:
        tok_s = fn()
        return {
            "path": label,
            "tok_s": round(tok_s, 1),
            "speedup_vs_eager": round(tok_s / baseline, 3) if baseline else None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures, not hide them
        return {"path": label, "tok_s": None, "speedup_vs_eager": None, "error": f"{type(exc).__name__}: {exc}"[:500]}


def bench_one(size: str, args: argparse.Namespace) -> dict[str, Any]:
    dtype = DTYPE_MAP[args.dtype]
    vocab = args.vocab
    model = build_model(size, dtype, args.device, vocab)
    input_ids = torch.randint(0, vocab, (args.batch, args.prompt), device=args.device)
    max_seq_len = args.prompt + args.tokens + args.warmup_tokens + 16
    rows: list[dict[str, Any]] = []

    needs_qwen35 = any(
        path in args.paths for path in ("hf-graph", "patched-eager", "patched-graph", "patched-compile")
    )
    if needs_qwen35:
        _, patch_model, unpatch_model = require_qwen35_triton()
    else:
        patch_model = unpatch_model = None

    if unpatch_model is not None:
        unpatch_model()

    eager = timed_decode(eager_loop(model, model.config, input_ids), args.tokens, args.warmup_tokens)
    if "eager" in args.paths:
        rows.append({"path": "eager", "tok_s": round(eager, 1), "speedup_vs_eager": 1.0, "error": None})

    if "compile" in args.paths:
        if unpatch_model is not None:
            unpatch_model()
        for mode in args.compile_modes:
            torch.compiler.reset()

            def run_compile(mode: str = mode) -> float:
                return timed_decode(
                    compiled_loop(model, model.config, input_ids, mode, max_seq_len),
                    args.tokens,
                    max(args.warmup_tokens, 8),
                )

            rows.append(
                measure(
                    f"compile-{mode}",
                    run_compile,
                    eager,
                )
            )

    if "hf-graph" in args.paths:
        assert unpatch_model is not None
        unpatch_model()

        def run_hf_graph() -> float:
            return timed_decode(graph_loop(model, input_ids, max_seq_len), args.tokens, args.warmup_tokens)

        rows.append(measure("hf-graph", run_hf_graph, eager))

    if "patched-eager" in args.paths:
        assert patch_model is not None and unpatch_model is not None
        patch_model(model)

        def run_patched_eager() -> float:
            return timed_decode(eager_loop(model, model.config, input_ids), args.tokens, args.warmup_tokens)

        rows.append(measure("patched-eager", run_patched_eager, eager))
        unpatch_model()

    if "patched-graph" in args.paths:
        assert patch_model is not None and unpatch_model is not None
        patch_model(model)

        def run_patched_graph() -> float:
            return timed_decode(graph_loop(model, input_ids, max_seq_len), args.tokens, args.warmup_tokens)

        rows.append(measure("patched-graph", run_patched_graph, eager))
        unpatch_model()

    if "patched-compile" in args.paths:
        assert patch_model is not None and unpatch_model is not None
        patch_model(model)
        for mode in args.compile_modes:
            torch.compiler.reset()

            def run_patched_compile(mode: str = mode) -> float:
                return timed_decode(
                    compiled_loop(model, model.config, input_ids, mode, max_seq_len),
                    args.tokens,
                    max(args.warmup_tokens, 8),
                )

            rows.append(measure(f"patched-compile-{mode}", run_patched_compile, eager))
        unpatch_model()

    gc.collect()
    torch.cuda.empty_cache()
    return {"size": size, "paths": rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    deltanet_fast_path = fast_path_status()
    if deltanet_fast_path != "fla+causal_conv1d" and not args.allow_slow_deltanet:
        raise SystemExit(
            "Qwen3.5 benchmark requires the DeltaNet fast path for fair numbers. "
            f"Found `{deltanet_fast_path}`; run `uv sync --extra qwen35` or pass "
            "`--allow-slow-deltanet` for a diagnostic fallback-only run."
        )

    return {
        "benchmark": "qwen3.5 decode attribution",
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "host": platform.node(),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "batch": args.batch,
        "prompt": args.prompt,
        "decode_tokens": args.tokens,
        "timed_region": "fixed prompt prefill plus generated decode tokens; compile/graph/custom interventions target the fixed-shape decode step",
        "warmup_tokens": args.warmup_tokens,
        "deltanet_fast_path": deltanet_fast_path,
        "sizes": args.sizes,
        "paths_requested": args.paths,
        "compile_modes": args.compile_modes,
        "results": [bench_one(size, args) for size in args.sizes],
    }


def format_table(record: dict[str, Any]) -> str:
    all_paths = []
    for result in record["results"]:
        for row in result["paths"]:
            if row["path"] not in all_paths:
                all_paths.append(row["path"])

    lines = [
        "# Qwen3.5 Decode Attribution",
        "",
        f"- device: **{record['device']}** on `{record['host']}`",
        f"- torch: `{record['torch']}` | dtype: `{record['dtype']}` | batch: {record['batch']}",
        f"- prompt: {record['prompt']} tokens | decode: {record['decode_tokens']} tokens",
        f"- DeltaNet fast path: `{record['deltanet_fast_path']}`",
        "",
        "| size | " + " | ".join(all_paths) + " |",
        "|---|" + "|".join(["--:"] * len(all_paths)) + "|",
    ]
    for result in record["results"]:
        by_path = {row["path"]: row for row in result["paths"]}
        cells = []
        for path in all_paths:
            row = by_path.get(path)
            if row is None:
                cells.append("")
            elif row["tok_s"] is None:
                cells.append("FAILED")
            elif path == "eager":
                cells.append(f"{row['tok_s']:.0f} tok/s")
            else:
                cells.append(f"{row['tok_s']:.0f} tok/s ({row['speedup_vs_eager']:.2f}x)")
        lines.append(f"| {result['size']} | " + " | ".join(cells) + " |")

    errors = [
        f"- {result['size']} {row['path']}: {row['error']}"
        for result in record["results"]
        for row in result["paths"]
        if row.get("error")
    ]
    if errors:
        lines += ["", "Errors:", *errors]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP))
    parser.add_argument("--sizes", nargs="+", default=["0.8B", "2B", "4B", "9B"], choices=list(QWEN35_CONFIGS))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--paths", nargs="+", default=DEFAULT_PATHS, choices=DEFAULT_PATHS)
    parser.add_argument("--compile-modes", nargs="+", default=["reduce-overhead", "max-autotune"])
    parser.add_argument(
        "--allow-slow-deltanet",
        action="store_true",
        help="Allow a diagnostic run without flash-linear-attention + causal-conv1d.",
    )
    parser.add_argument("--json", default=None, help="Optional path for raw JSON results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    record = run(args)
    record["wall_s"] = round(time.time() - t0, 1)
    print(format_table(record), flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"\nwrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
