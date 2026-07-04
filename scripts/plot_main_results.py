"""Render the README main-results figure from the result artifacts.

Reads the 9B attribution ablations and the Qwen3-TTS E2E runs, and draws the
measured value of every path (no derived ratios). Output: assets/main_results.svg.

Run: uv run --group plot python scripts/plot_main_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from tueplots import bundles

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "assets" / "main_results.svg"

# Ladder rows, top to bottom. None marks a path a case does not measure.
ROWS = ["Eager", "Kernels only", "Graph only", "Kernels + graph", "Compile only", "Kernels + compile"]
QWEN35_PATHS = {
    "Eager": "eager",
    "Kernels only": "patched-eager",
    "Graph only": "hf-graph",
    "Kernels + graph": "patched-graph",
    "Compile only": "compile-max-autotune",
    "Kernels + compile": "patched-compile-max-autotune",
}
TTS_LABELS = {
    "Eager": "repo-base",
    "Kernels only": None,  # upstream ships its Triton patches only inside the graph runner
    "Graph only": "repo-faster",
    "Kernels + graph": "repo-hybrid",
    "Compile only": "repo-compile-max-autotune",
    "Kernels + compile": "repo-hybrid-compile-max-autotune",
}
KERNEL_ROWS = {"Kernels only", "Kernels + graph", "Kernels + compile"}

# Categorical slots 1-2 and chart chrome from the dataviz reference palette.
COLOR_FRAMEWORK = "#2a78d6"
COLOR_KERNELS = "#1baf7a"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def qwen35_values(name: str) -> dict[str, float]:
    record = json.loads((RESULTS / name).read_text())
    by_path = {row["path"]: row["tok_s"] for row in record["results"][0]["paths"]}
    return {row: by_path[path] for row, path in QWEN35_PATHS.items()}


def tts_values(*names: str) -> dict[str, float | None]:
    by_label = {}
    for name in names:
        for row in json.loads((RESULTS / name).read_text()):
            by_label[row["label"]] = row["mean_ms"] * 1000 / row["audio"]
    return {row: by_label[label] if label else None for row, label in TTS_LABELS.items()}


def draw(ax, values: dict[str, float | None], title: str, fmt: str) -> None:
    ys = range(len(ROWS) - 1, -1, -1)  # Eager on top
    for y, row in zip(ys, ROWS):
        value = values[row]
        if value is None:
            ax.text(0, y, "  n/a", va="center", ha="left", color=INK_MUTED, fontsize=6)
            continue
        color = COLOR_KERNELS if row in KERNEL_ROWS else COLOR_FRAMEWORK
        ax.barh(y, value, height=0.62, color=color, zorder=3)
        ax.text(value, y, "  " + fmt.format(value), va="center", ha="left", color=INK_SECONDARY, fontsize=6)
    ax.set_yticks(list(ys), ROWS)
    ax.set_xlim(0, max(v for v in values.values() if v is not None) * 1.22)
    ax.set_title(title)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.5, zorder=0)
    ax.tick_params(colors=INK_MUTED, labelcolor="#0b0b0b")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)


def main() -> None:
    plt.rcParams.update(bundles.neurips2024(usetex=False, family="sans-serif", rel_width=1.3, nrows=2, ncols=2))
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # ships with matplotlib; avoids findfont fallback noise
    fig, axes = plt.subplots(2, 2, sharey="row")

    draw(axes[0][0], qwen35_values("qwen35_ablation_h100.json"), "Qwen3.5 9B decode, H100", "{:.0f}")
    draw(axes[0][1], qwen35_values("qwen35_ablation_a100.json"), "Qwen3.5 9B decode, A100", "{:.0f}")
    draw(
        axes[1][0],
        tts_values("qwen3_tts_h100_reduce.json", "qwen3_tts_h100_max_autotune.json"),
        "Qwen3-TTS E2E, H100",
        "{:.2f}",
    )
    draw(
        axes[1][1],
        tts_values("qwen3_tts_a100_reduce.json", "qwen3_tts_a100_max_autotune.json"),
        "Qwen3-TTS E2E, A100",
        "{:.2f}",
    )

    for ax in axes[0]:
        ax.set_xlabel("tok/s (higher is better)")
    for ax in axes[1]:
        ax.set_xlabel("ms per 1k output samples (lower is better)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FRAMEWORK, label="framework only"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_KERNELS, label="with custom kernels"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncols=2, frameon=False)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, facecolor="#ffffff", bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
