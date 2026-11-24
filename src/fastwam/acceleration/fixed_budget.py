"""Fixed-budget whole-action-block residual reuse and CUDA Graph execution."""
import time

import torch

from .schedule import LAYERS, STEPS, ActionSchedule
from ..models.wan22.mot import flash_attention


def _map_tensors(value, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, (tuple, list)):
        return type(value)(_map_tensors(item, transform) for item in value)
    raise TypeError(f"Unsupported graph input: {type(value).__name__}")


def _copy_tensors(target, source):
    if isinstance(target, torch.Tensor):
        target.copy_(source)
    else:
        for a, b in zip(target, source):
            _copy_tensors(a, b)


class CUDAGraphRunner:
    """Capture per input signature and copy all current inputs before replay."""

    def __init__(self, function, device):
        self.function = function
        self.device = torch.device(device)
        self.entries = {}
        self.capture_seconds = 0.0

    def __call__(self, *args):
        key = repr(_map_tensors(args, lambda t: (t.shape, t.dtype, t.device, t.stride())))
        with torch.cuda.device(self.device):
            if key not in self.entries:
                started = time.perf_counter()
                static = _map_tensors(args, lambda t: t.clone(memory_format=torch.preserve_format))
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                # Each distinct mask specializes the same Python step code.
                # Raise the Dynamo cache limit locally during graph preparation.
                from torch._dynamo import config as dynamo_config

                with dynamo_config.patch(
                    cache_size_limit=max(256, dynamo_config.cache_size_limit),
                    accumulated_cache_size_limit=max(1024, dynamo_config.accumulated_cache_size_limit),
                ):
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            self.function(*static)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        result = self.function(*static)
                    torch.cuda.synchronize()
                self.entries[key] = static, graph, result
                self.capture_seconds += time.perf_counter() - started
            static, graph, result = self.entries[key]
            _copy_tensors(static, args)
            graph.replay()
            return result


def _compile(function):
    return torch.compile(function, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})


class FixedBudgetRuntime:
    """Apply a frozen mask for one complete denoising loop per plan."""

    def __init__(self, model, schedule: ActionSchedule, backend: str = "graph"):
        if model.mot.num_layers != LAYERS:
            raise ValueError(f"Fixed action budgets require {LAYERS} layers")
        if backend not in ("eager", "graph"):
            raise ValueError("Fixed action budget backend must be eager or graph")
        if backend == "graph" and torch.device(model.device).type != "cuda":
            raise ValueError("The fixed budget graph backend requires CUDA")
        self.model = model
        self.schedule = schedule
        self.backend = backend
        self.steps = {}
        for mask in sorted(set(schedule.mask)):
            if schedule.calls == STEPS * LAYERS:
                break
            step = self._make_step(mask)
            self.steps[mask] = _compile(step) if backend == "graph" else step
        self.dense_step = None
        if schedule.calls == STEPS * LAYERS:
            step = model._denoise_action_with_video_cache
            self.dense_step = _compile(step) if backend == "graph" else step
        self.run = self._run
        self._prefill = model.mot.prefill_video_cache_tensor
        self._video_runner = None
        self._action_runner = None
        self._prompt_cache = {}
        if backend == "graph":
            self._action_runner = CUDAGraphRunner(self._run, model.device)
            self._video_runner = CUDAGraphRunner(_compile(self._prefill_positional), model.device)
            self.run = self._action_runner

    @property
    def capture_seconds(self) -> float:
        return sum(x.capture_seconds for x in (self._video_runner, self._action_runner) if x is not None)

    def encode_prompt(self, prompt):
        key = prompt if isinstance(prompt, str) else tuple(prompt)
        if key not in self._prompt_cache:
            self._prompt_cache[key] = self.model.encode_prompt(prompt)
        return self._prompt_cache[key]

    def _prefill_positional(self, tokens, freqs, t_mod, context, context_mask, attention_mask):
        return self._prefill(video_tokens=tokens, video_freqs=freqs, video_t_mod=t_mod,
                             video_context=context, video_context_mask=context_mask,
                             video_attention_mask=attention_mask)

    def prefill(self, *, video_tokens, video_freqs, video_t_mod, video_context,
                video_context_mask, video_attention_mask):
        runner = self._video_runner or self._prefill_positional
        return runner(video_tokens, video_freqs, video_t_mod, video_context,
                      video_context_mask, video_attention_mask)

    def _make_step(self, row):
        mot, expert = self.model.mot, self.model.action_expert

        def step(latents, timestep, context, context_mask, video_k, video_v, attention_mask, cache):
            x, _, t_mod, ctx, ctx_mask, freqs = expert.prepare(
                action_tokens=latents, timestep=timestep, context=context, context_mask=context_mask)
            updated = []
            for layer, skip in enumerate(row):
                if skip:
                    residual = cache[layer]
                    x = x + residual
                else:
                    before = x
                    block = expert.blocks[layer]
                    q, k, v, residual_x, gate, shift, scale, mlp_gate, _ = mot._build_expert_attention_io(
                        expert=expert, block=block, x=x, freqs=freqs, t_mod=t_mod)
                    mixed = flash_attention(
                        q=q, k=torch.cat([video_k[layer], k], dim=1), v=torch.cat([video_v[layer], v], dim=1),
                        num_heads=mot.num_heads, ctx_mask=attention_mask.to(device=q.device))
                    x = mot._apply_expert_post_block_tensor(
                        block=block, residual_x=residual_x, mixed_attn_out=mixed, gate_msa=gate,
                        shift_mlp=shift, scale_mlp=scale, gate_mlp=mlp_gate, context=ctx, context_mask=ctx_mask)
                    residual = x - before
                updated.append(residual)
            return expert.post(x), updated

        return step

    def _run(self, latents, context, context_mask, video_k, video_v, attention_mask, timesteps, deltas):
        if len(timesteps) != STEPS or len(deltas) != STEPS:
            raise ValueError(f"Fixed action budgets require {STEPS} denoising steps")
        cache = [None] * LAYERS
        for row, t, delta in zip(self.schedule.mask, timesteps, deltas):
            timestep = t.unsqueeze(0).to(dtype=latents.dtype, device=latents.device)
            if self.dense_step is None:
                prediction, cache = self.steps[row](
                    latents, timestep, context, context_mask, video_k, video_v, attention_mask, cache)
            else:
                prediction = self.dense_step(
                    latents_action=latents, timestep_action=timestep, context=context, context_mask=context_mask,
                    video_cache_k=video_k, video_cache_v=video_v, action_attention_mask=attention_mask)
            latents = self.model.infer_action_scheduler.step(prediction, delta, latents)
        return latents
