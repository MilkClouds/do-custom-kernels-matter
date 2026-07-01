"""CUDA-graph capture for the Qwen3.5 autoregressive decode step.

The fused Triton kernels cut per-op launch *count*, but a Qwen3.5 decode step
still issues hundreds of launches (one set per layer) and the CPU dispatch of
those launches — not GPU compute — is the decode-time bottleneck. A CUDA graph
captures the entire decode step's launch sequence once and replays it as a
single submission, so per-token CPU overhead drops to ~zero.

This is a *separate* lever from kernel fusion. Fusion shrinks each op; the
graph removes the dispatch cost of whatever launches remain. They compose.

Capture has one hard precondition: every decode step must touch identical
tensor shapes *and addresses*. That requires:

- a **static KV / state cache** — HF's :class:`~transformers.StaticCache`,
  which for a hybrid model like Qwen3.5 builds a fixed-``max_cache_len``
  ``StaticLayer`` for each full-attention layer and a fixed-size
  ``LinearAttentionLayer`` (conv + recurrent state) for each gated-delta-net
  layer. Both mark their tensors with ``mark_static_address``.
- **static input buffers** — a 1-token ``input_ids`` and a 1-element
  ``cache_position``, pre-allocated once and mutated in place between replays.

Prefill is *not* graphed: it is variable-length and runs once. Only the
fixed-shape decode step is captured.

:class:`GraphedDecoder` wires this together behind a small ``generate`` API.
"""

from __future__ import annotations

import torch

# The decode step is graphed; prefill is not. Warmup iterations are run on a
# side stream before capture so lazy allocations / autotune settle first.
_DEFAULT_WARMUP = 3


def _qwen35_static_cache(config, max_cache_len: int):
    """Build a hybrid :class:`StaticCache` sized for ``max_cache_len`` tokens.

    ``StaticCache`` reads ``config.layer_types`` and gives every
    ``full_attention`` layer a static KV buffer and every ``linear_attention``
    layer a fixed-size conv+recurrent buffer — exactly the hybrid layout
    Qwen3.5 needs, all CUDA-graph-capturable.
    """
    from transformers import StaticCache

    return StaticCache(config=config, max_cache_len=max_cache_len)


class GraphedDecoder:
    """Greedy decoder for a HF Qwen3.5 model with a CUDA-graphed decode step.

    Usage::

        from transformers import AutoModelForCausalLM
        from qwen35_triton import patch_model, GraphedDecoder

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B").cuda().eval()
        patch_model(model)                       # fused Triton kernels
        decoder = GraphedDecoder(model, max_seq_len=4096)
        out_ids = decoder.generate(input_ids, max_new_tokens=256)

    ``patch_model`` (Triton kernels) and the CUDA graph are independent and
    compose — apply the patch first for the full speedup.

    The first ``generate`` call after construction captures the graph (one-off
    cost); subsequent tokens replay it. A capture is reusable across
    ``generate`` calls as long as the batch size is unchanged.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_seq_len: int = 4096,
        warmup: int = _DEFAULT_WARMUP,
    ) -> None:
        """Args:
        model: a loaded HF Qwen3.5 causal-LM, already on a CUDA device.
        max_seq_len: KV/state cache capacity. Prompt length + max_new_tokens
            must not exceed this — the static cache cannot grow.
        warmup: decode-step iterations run before capture so lazy
            allocations and Triton autotune settle.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("GraphedDecoder requires CUDA")
        self.model = model
        self.config = model.config.get_text_config(decoder=True)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.max_seq_len = int(max_seq_len)
        self.warmup = int(warmup)

        # Capture state — lazily built on the first generate() of a given batch.
        self._graph: torch.cuda.CUDAGraph | None = None
        self._cache = None
        self._static_input: torch.Tensor | None = None
        self._static_pos: torch.Tensor | None = None
        self._static_logits: torch.Tensor | None = None
        self._captured_batch: int | None = None

    # ── internals ────────────────────────────────────────────────────────────

    def _decode_forward(self) -> torch.Tensor:
        """One decode step over the static buffers; returns the last-token logits."""
        out = self.model(
            self._static_input,
            past_key_values=self._cache,
            use_cache=True,
            cache_position=self._static_pos,
        )
        return out.logits[:, -1, :]

    def _capture(self, batch_size: int) -> None:
        """Warm up then capture the decode step into a CUDA graph.

        Assumes ``self._cache`` already holds the prompt (prefill done) and the
        static input buffers are allocated. The capture records the launch
        sequence for a single decode step; ``replay`` reruns it.
        """
        # Warmup on a side stream so capture sees a settled allocator.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            with torch.no_grad():
                for _ in range(self.warmup):
                    self._decode_forward()
        torch.cuda.current_stream().wait_stream(side)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            with torch.no_grad():
                self._static_logits = self._decode_forward()
        self._captured_batch = batch_size

    # ── public API ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy-decode ``max_new_tokens`` tokens after ``input_ids``.

        Prefill runs as a normal (un-graphed) forward; the decode loop replays
        the captured CUDA graph one token at a time.

        Args:
            input_ids: ``[batch, prompt_len]`` prompt token ids on the model's
                device.
            max_new_tokens: number of tokens to generate.
            eos_token_id: if set, generation stops early once *every* sequence
                in the batch has emitted it.

        Returns:
            ``[batch, prompt_len + generated]`` token ids (prompt included).
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, prompt_len], got {tuple(input_ids.shape)}")
        batch_size, prompt_len = input_ids.shape
        if prompt_len + max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt_len ({prompt_len}) + max_new_tokens ({max_new_tokens}) exceeds "
                f"max_seq_len ({self.max_seq_len}); construct GraphedDecoder with a larger max_seq_len"
            )
        input_ids = input_ids.to(self.device)

        # A capture is batch-size specific; rebuild if the batch changed.
        if self._captured_batch != batch_size:
            self._graph = None
            self._cache = None

        # The static cache is built once and reused. Its layer tensors keep a
        # fixed address (StaticLayer / LinearAttentionLayer mark_static_address),
        # so the captured graph stays valid across generate() calls — between
        # calls we only reset() the contents in place, never reallocate.
        if self._cache is None:
            self._cache = _qwen35_static_cache(self.config, self.max_seq_len)
        else:
            self._cache.reset()

        prefill_pos = torch.arange(prompt_len, device=self.device)
        prefill_out = self.model(
            input_ids,
            past_key_values=self._cache,
            use_cache=True,
            cache_position=prefill_pos,
        )
        next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [batch, 1]

        generated = [next_token]
        # Static decode buffers — fixed address, mutated in place between replays.
        if self._static_input is None or self._static_input.shape[0] != batch_size:
            self._static_input = torch.zeros(batch_size, 1, dtype=input_ids.dtype, device=self.device)
            self._static_pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self._static_input.copy_(next_token)
        self._static_pos.fill_(prompt_len)

        # Capture once. Warmup + capture run decode steps that advance the
        # cache (including the cumulative DeltaNet recurrent state), so the
        # cache is reset() and re-prefilled in place afterwards — same tensors,
        # same addresses, contents rewound to the post-prefill state.
        if self._graph is None:
            self._capture(batch_size)
            self._cache.reset()
            self.model(input_ids, past_key_values=self._cache, use_cache=True, cache_position=prefill_pos)
            self._static_input.copy_(next_token)
            self._static_pos.fill_(prompt_len)

        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for _ in range(max_new_tokens - 1):
            self._graph.replay()
            next_token = self._static_logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if eos_token_id is not None:
                done |= next_token.squeeze(-1) == eos_token_id
                if bool(done.all()):
                    break
            # Advance the static buffers in place for the next replay.
            self._static_input.copy_(next_token)
            self._static_pos.add_(1)

        return torch.cat([input_ids, *generated], dim=1)

    def reset(self) -> None:
        """Drop the captured graph and cache. The next ``generate`` recaptures."""
        self._graph = None
        self._cache = None
        self._captured_batch = None
