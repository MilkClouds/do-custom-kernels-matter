import gc
import json
import os
import traceback

import numpy as np
import torch

ATTN = os.getenv("ATTN", "sdpa")
MNT = int(os.getenv("MNT", "256"))
WARMUP = int(os.getenv("WARMUP", "3"))
REPEAT = int(os.getenv("REPEAT", "5"))
COMPILE_MODE = os.getenv("COMPILE_MODE", "reduce-overhead")
TEXT = os.getenv(
    "TEXT",
    "Hello, this is a fair decode benchmark of the text to speech system.",
)
OUT = os.getenv("OUT", "decompose_results.json")
ROWS = {row.strip() for row in os.getenv("ROWS", "base,faster,compile,hybrid").split(",") if row.strip()}


def audio_len(out):
    if isinstance(out, dict):
        audio = out.get("audio")
    elif isinstance(out, (tuple, list)):
        audio = out[0]
    else:
        audio = out
    while isinstance(audio, (tuple, list)):
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu()
    return int(np.asarray(audio).reshape(-1).shape[0])


def timed(fn, warmup, repeat):
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    values = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))

    arr = np.asarray(values)
    return float(arr.mean()), float(arr.min()), audio_len(out)


def gen(tts):
    return tts.generate_custom_voice(
        text=TEXT,
        language="english",
        speaker="vivian",
        max_new_tokens=MNT,
        do_sample=False,
    )


def run(label, fn):
    try:
        row = fn()
        print("ok", row, flush=True)
        return row
    except Exception:
        traceback.print_exc()
        print(f"FAIL {label}", flush=True)
        return None


def run_repo(label, module_name, class_name):
    def _run():
        module = __import__(
            f"qwen3_tts_triton.models.{module_name}",
            fromlist=[class_name],
        )
        runner = getattr(module, class_name)()
        runner.load_model()
        # FasterQwen3TTS bakes predictor sampling into the captured graph. The
        # public runner's greedy=True does not update that graph object, so force
        # it before the first generate/capture for a deterministic comparison.
        predictor_graph = getattr(getattr(runner, "model", None), "predictor_graph", None)
        if predictor_graph is not None:
            predictor_graph.do_sample = False
        mean_ms, min_ms, audio = timed(
            lambda: runner.generate(
                text=TEXT,
                language="English",
                max_new_tokens=MNT,
                greedy=True,
            ),
            WARMUP,
            REPEAT,
        )
        if hasattr(runner, "unload_model"):
            runner.unload_model()
        gc.collect()
        torch.cuda.empty_cache()
        return {"label": label, "attn": "repo-default", "mean_ms": mean_ms, "min_ms": min_ms, "audio": audio}

    return run(label, _run)


def run_compiled():
    def _run():
        from kernelcheck.compiled_qwen3_tts import CompiledFasterRunner

        runner = CompiledFasterRunner(compile_mode=COMPILE_MODE, attn_implementation=ATTN)
        runner.load_model()
        mean_ms, min_ms, audio = timed(
            lambda: runner.generate(
                text=TEXT,
                language="English",
                max_new_tokens=MNT,
                greedy=True,
            ),
            WARMUP,
            REPEAT,
        )
        runner.unload_model()
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "label": f"repo-compile-{COMPILE_MODE}",
            "attn": ATTN,
            "mean_ms": mean_ms,
            "min_ms": min_ms,
            "audio": audio,
        }

    return run(f"repo-compile-{COMPILE_MODE}", _run)


def ms_per_1k(row):
    return row["mean_ms"] * 1000 / row["audio"]


def main():
    print(
        f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} attn={ATTN} mnt={MNT}",
        flush=True,
    )
    rows = []
    if "base" in ROWS:
        rows.append(run_repo("repo-base", "base_runner", "BaseRunner"))
    if "faster" in ROWS:
        rows.append(run_repo("repo-faster", "faster_runner", "FasterRunner"))
    if "compile" in ROWS and os.getenv("INCLUDE_COMPILE", "1") == "1":
        rows.append(run_compiled())
    if "hybrid" in ROWS:
        rows.append(run_repo("repo-hybrid", "triton_faster_runner", "TritonFasterRunner"))
    rows = [r for r in rows if r]

    if not rows:
        raise SystemExit("No rows selected")

    print("\nlabel                   mean_ms  audio  ms/1k_samples", flush=True)
    for row in rows:
        print(
            f"{row['label']:<24} {row['mean_ms']:>7.0f}  {row['audio']:>6}  {ms_per_1k(row):>13.2f}",
            flush=True,
        )

    by_label = {row["label"]: row for row in rows}
    faster = by_label.get("repo-faster")
    hybrid = by_label.get("repo-hybrid")
    if faster and hybrid:
        audio_delta = abs(hybrid["audio"] - faster["audio"]) / faster["audio"]
        marginal = faster["mean_ms"] / hybrid["mean_ms"]
        print(
            f"\nFaster -> Hybrid kernel marginal: {marginal:.2f}x "
            f"(audio_delta={audio_delta:.1%}, valid={audio_delta <= 0.05})",
            flush=True,
        )
    compile_row = next((row for row in rows if row["label"].startswith("repo-compile-")), None)
    if compile_row and hybrid:
        wall = hybrid["mean_ms"] / compile_row["mean_ms"]
        sample_norm = ms_per_1k(hybrid) / ms_per_1k(compile_row)
        print(
            f"torch.compile -> Hybrid: {wall:.2f}x faster wall-clock, {sample_norm:.2f}x faster per generated sample",
            flush=True,
        )

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
