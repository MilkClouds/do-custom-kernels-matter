"""torch.compile runner for the same Qwen3-TTS boundaries as faster-qwen3-tts."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch


def _to_numpy(audio: Any) -> np.ndarray:
    if isinstance(audio, list):
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        return audio.squeeze().detach().cpu().float().numpy()
    return np.asarray(audio, dtype=np.float32)


class CompiledPredictorGraph:
    """PredictorGraph-compatible wrapper using torch.compile instead of CUDAGraph."""

    def __init__(self, *args: Any, compile_mode: str = "reduce-overhead", **kwargs: Any) -> None:
        from faster_qwen3_tts.predictor_graph import PredictorGraph

        self._graph = PredictorGraph(*args, **kwargs)
        self.compile_mode = compile_mode
        self._compiled_loop = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)

    def _full_loop_functional(self) -> torch.Tensor:
        """Predict the 15 codebooks without mutating the graph output buffer."""
        from faster_qwen3_tts.sampling import sample_logits

        g = self._graph
        h = g.small_to_mtp(g.input_buf)
        out = g.pred_model(
            inputs_embeds=h,
            attention_mask=g.prefill_attn,
            past_key_values=g.static_cache,
            cache_position=g.prefill_cache_pos,
            use_cache=True,
        )
        h = out.last_hidden_state

        logits = g.lm_heads[0](h[:, -1:, :])
        tok = sample_logits(
            logits[:, 0, :],
            temperature=g.temperature,
            top_k=g.top_k,
            top_p=g.top_p,
            do_sample=g.do_sample,
        )
        tokens = [tok]

        for cb_idx in range(1, g.num_codebooks):
            emb = g.codec_embeds[cb_idx - 1](tok.unsqueeze(0))
            emb = g.small_to_mtp(emb)
            out = g.pred_model(
                inputs_embeds=emb,
                attention_mask=g.decode_attn[cb_idx - 1],
                past_key_values=g.static_cache,
                cache_position=g.decode_cache_positions[cb_idx - 1],
                use_cache=True,
            )
            h = out.last_hidden_state
            logits = g.lm_heads[cb_idx](h[:, -1:, :])
            tok = sample_logits(
                logits[:, 0, :],
                temperature=g.temperature,
                top_k=g.top_k,
                top_p=g.top_p,
                do_sample=g.do_sample,
            )
            tokens.append(tok)

        return torch.cat(tokens, dim=0)

    @torch.inference_mode()
    def capture(self, num_warmup: int = 3) -> None:
        if self._compiled_loop is not None:
            return
        print(f"Warming up compiled predictor ({num_warmup} runs)...")
        self._graph._init_cache_layers()
        self._graph._build_attention_masks()
        self._compiled_loop = torch.compile(
            self._full_loop_functional,
            mode=self.compile_mode,
            fullgraph=False,
        )
        for _ in range(num_warmup):
            self._graph.static_cache.reset()
            torch.compiler.cudagraph_mark_step_begin()
            self._compiled_loop()
        torch.cuda.synchronize()
        self._graph.captured = True
        print("Compiled predictor ready!")

    @torch.inference_mode()
    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        if self._compiled_loop is None:
            self.capture()
        self._graph.input_buf.copy_(pred_input)
        self._graph.static_cache.reset()
        torch.compiler.cudagraph_mark_step_begin()
        assert self._compiled_loop is not None
        tokens = self._compiled_loop()
        return tokens.clone()


class CompiledTalkerGraph:
    """TalkerGraph-compatible wrapper using torch.compile instead of CUDAGraph."""

    def __init__(self, *args: Any, compile_mode: str = "reduce-overhead", **kwargs: Any) -> None:
        from faster_qwen3_tts.talker_graph import TalkerGraph

        self._graph = TalkerGraph(*args, **kwargs)
        self.compile_mode = compile_mode
        self._compiled_decode = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)

    def _decode_step_functional(self) -> torch.Tensor:
        """Decode one talker token without mutating the graph output buffer."""
        g = self._graph
        out = g.model(
            inputs_embeds=g.input_buf,
            attention_mask=g.attn_mask,
            past_key_values=g.static_cache,
            cache_position=g.cache_position,
            position_ids=g.position_ids,
            use_cache=True,
        )
        return out.last_hidden_state

    @torch.inference_mode()
    def capture(self, prefill_len: int = 100, num_warmup: int = 3) -> None:
        if self._compiled_decode is not None:
            return
        print(f"Warming up compiled talker ({num_warmup} runs)...")
        self._graph._init_cache_layers()
        self._graph._build_attention_masks()
        self._graph.cache_position[0] = prefill_len
        self._graph._set_attention_mask(prefill_len)
        self._compiled_decode = torch.compile(
            self._decode_step_functional,
            mode=self.compile_mode,
            fullgraph=False,
        )
        for _ in range(num_warmup):
            torch.compiler.cudagraph_mark_step_begin()
            self._compiled_decode()
        torch.cuda.synchronize()
        self._graph.captured = True
        print("Compiled talker ready!")

    def reset(self, prefill_len: int) -> None:
        self._graph.reset(prefill_len)

    def prefill_kv(self, past_key_values: Any) -> int:
        return self._graph.prefill_kv(past_key_values)

    def set_generation_state(self, attention_mask: torch.Tensor, rope_deltas: torch.Tensor | None) -> None:
        self._graph.set_generation_state(attention_mask, rope_deltas)

    @torch.inference_mode()
    def run(self, input_embeds: torch.Tensor, position: int) -> torch.Tensor:
        if self._compiled_decode is None:
            self.capture(prefill_len=position)
        self._graph.input_buf.copy_(input_embeds)
        self._graph.cache_position[0] = position
        self._graph._set_attention_mask(position)
        delta = self._graph.rope_deltas + self._graph.cache_position[0].to(self._graph.rope_deltas.dtype)
        self._graph.position_ids.copy_(delta.unsqueeze(0).expand(3, -1, -1))
        torch.compiler.cudagraph_mark_step_begin()
        assert self._compiled_decode is not None
        return self._compiled_decode().clone()


def create_compiled_faster_qwen3_tts(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    max_seq_len: int = 2048,
    compile_mode: str = "reduce-overhead",
    do_sample: bool = False,
):
    """Create a FasterQwen3TTS-compatible object with compiled decode kernels."""
    from faster_qwen3_tts.model import FasterQwen3TTS, suppress_flash_attn_warning

    with suppress_flash_attn_warning():
        from qwen_tts import Qwen3TTSModel

    base_model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )

    talker = base_model.model.talker
    talker_config = base_model.model.config.talker_config
    predictor = talker.code_predictor
    pred_config = predictor.model.config

    predictor_graph = CompiledPredictorGraph(
        predictor,
        pred_config,
        talker_config.hidden_size,
        device=device,
        dtype=dtype,
        do_sample=do_sample,
        top_k=50,
        temperature=0.9,
        compile_mode=compile_mode,
    )
    talker_graph = CompiledTalkerGraph(
        talker.model,
        talker_config,
        device=device,
        dtype=dtype,
        max_seq_len=max_seq_len,
        compile_mode=compile_mode,
    )
    return FasterQwen3TTS(
        base_model=base_model,
        predictor_graph=predictor_graph,
        talker_graph=talker_graph,
        device=device,
        dtype=dtype,
        max_seq_len=max_seq_len,
    )


class CompiledFasterRunner:
    """Small runner matching qwen3_tts_triton's Base/Faster/Hybrid interface."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        compile_mode: str = "reduce-overhead",
        attn_implementation: str = "sdpa",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.compile_mode = compile_mode
        self.attn_implementation = attn_implementation
        self.model = None

    def load_model(self) -> None:
        self.model = create_compiled_faster_qwen3_tts(
            self.model_id,
            device=self.device,
            dtype=self.dtype,
            compile_mode=self.compile_mode,
            attn_implementation=self.attn_implementation,
            do_sample=False,
        )

    def generate(
        self,
        text: str,
        language: str = "en",
        speaker: str = "vivian",
        *,
        max_new_tokens: int = 2048,
        greedy: bool = False,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        start = time.perf_counter()
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language.lower(),
            speaker=speaker,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
        )
        torch.cuda.synchronize()
        return {
            "audio": _to_numpy(wavs),
            "sample_rate": sr,
            "time_s": time.perf_counter() - start,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def unload_model(self) -> None:
        self.model = None
        torch.cuda.empty_cache()
