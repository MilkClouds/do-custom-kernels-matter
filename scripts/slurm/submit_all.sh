#!/usr/bin/env bash
set -euo pipefail

THIS=${THIS:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
QWEN3_TTS=${QWEN3_TTS:-"$THIS/../qwen3-tts-triton"}
HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}
UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}
RUFF_CACHE_DIR=${RUFF_CACHE_DIR:-/tmp/ruff-cache}

if [[ ! -x "$QWEN3_TTS/.venv/bin/python" ]]; then
  echo "Qwen3-TTS benchmarks require QWEN3_TTS to point at a qwen3-tts-triton checkout with .venv." >&2
  echo "Example: QWEN3_TTS=/path/to/qwen3-tts-triton bash scripts/slurm/submit_all.sh" >&2
  exit 1
fi

submit_qwen35_decode() {
  local partition=$1
  local out_json=$2
  local mem=120G
  local exclude_args=()
  if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
    exclude_args=(--exclude="$SLURM_EXCLUDE")
  fi
  if [[ "$partition" == "a100" ]]; then
    mem=64G
  fi
  sbatch -p "$partition" "${exclude_args[@]}" --gres=gpu:1 --cpus-per-task=8 --mem="$mem" -t 4:00:00 \
    --job-name="qwen35-${partition}" --output="/tmp/qwen35_${partition}_%j.log" \
    --wrap="bash -lc 'cd $THIS && export HF_HOME=$HF_HOME PYTHONPATH=$THIS/src \
      UV_CACHE_DIR=$UV_CACHE_DIR RUFF_CACHE_DIR=$RUFF_CACHE_DIR && \
      uv run --extra qwen35 python -u scripts/qwen35/bench_decode.py \
        --sizes 0.8B 2B 4B 9B 27B \
        --paths eager compile patched-graph \
        --json $out_json'"
}

submit_qwen35_ablation() {
  local partition=$1
  local out_json=$2
  local mem=80G
  local exclude_args=()
  if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
    exclude_args=(--exclude="$SLURM_EXCLUDE")
  fi
  if [[ "$partition" == "a100" ]]; then
    mem=64G
  fi
  sbatch -p "$partition" "${exclude_args[@]}" --gres=gpu:1 --cpus-per-task=8 --mem="$mem" -t 2:00:00 \
    --job-name="qwen35-abl-${partition}" --output="/tmp/qwen35_ablation_${partition}_%j.log" \
    --wrap="bash -lc 'cd $THIS && export HF_HOME=$HF_HOME PYTHONPATH=$THIS/src \
      UV_CACHE_DIR=$UV_CACHE_DIR RUFF_CACHE_DIR=$RUFF_CACHE_DIR && \
      uv run --extra qwen35 python -u scripts/qwen35/bench_decode.py \
        --sizes 9B \
        --paths eager hf-graph patched-eager patched-graph \
        --json $out_json'"
}

submit_qwen3_tts_reduce() {
  local partition=$1
  local out_json=$2
  local mem=80G
  local exclude_args=()
  if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
    exclude_args=(--exclude="$SLURM_EXCLUDE")
  fi
  if [[ "$partition" == "a100" ]]; then
    mem=64G
  fi
  sbatch -p "$partition" "${exclude_args[@]}" --gres=gpu:1 --cpus-per-task=8 --mem="$mem" -t 2:00:00 \
    --job-name="qwen3tts-${partition}" --output="/tmp/qwen3tts_${partition}_%j.log" \
    --wrap="bash -lc 'cd $THIS && export HF_HOME=$HF_HOME && \
      export PYTHONPATH=$THIS/src:$QWEN3_TTS/src:$QWEN3_TTS && \
      export MNT=256 WARMUP=2 REPEAT=3 ROWS=base,faster,compile,hybrid ATTN=sdpa \
      INCLUDE_COMPILE=1 COMPILE_MODE=reduce-overhead OUT=$out_json && \
      exec $QWEN3_TTS/.venv/bin/python -u $THIS/scripts/qwen3_tts/bench_e2e.py'"
}

submit_qwen3_tts_max() {
  local partition=$1
  local out_json=$2
  local mem=80G
  local exclude_args=()
  if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
    exclude_args=(--exclude="$SLURM_EXCLUDE")
  fi
  if [[ "$partition" == "a100" ]]; then
    mem=64G
  fi
  sbatch -p "$partition" "${exclude_args[@]}" --gres=gpu:1 --cpus-per-task=8 --mem="$mem" -t 2:00:00 \
    --job-name="qwen3tts-max-${partition}" --output="/tmp/qwen3tts_max_${partition}_%j.log" \
    --wrap="bash -lc 'cd $THIS && export HF_HOME=$HF_HOME && \
      export PYTHONPATH=$THIS/src:$QWEN3_TTS/src:$QWEN3_TTS && \
      export MNT=256 WARMUP=1 REPEAT=3 ROWS=compile ATTN=sdpa \
      INCLUDE_COMPILE=1 COMPILE_MODE=max-autotune OUT=$out_json && \
      exec $QWEN3_TTS/.venv/bin/python -u $THIS/scripts/qwen3_tts/bench_e2e.py'"
}

submit_qwen35_decode h100 "$THIS/results/qwen35_decode_h100.json"
submit_qwen35_decode a100 "$THIS/results/qwen35_decode_a100.json"
submit_qwen35_ablation h100 "$THIS/results/qwen35_ablation_h100.json"
submit_qwen35_ablation a100 "$THIS/results/qwen35_ablation_a100.json"
submit_qwen3_tts_reduce h100 "$THIS/results/qwen3_tts_h100_reduce.json"
submit_qwen3_tts_reduce a100 "$THIS/results/qwen3_tts_a100_reduce.json"
submit_qwen3_tts_max h100 "$THIS/results/qwen3_tts_h100_max_autotune.json"
submit_qwen3_tts_max a100 "$THIS/results/qwen3_tts_a100_max_autotune.json"
