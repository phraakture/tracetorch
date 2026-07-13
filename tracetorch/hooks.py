"""PyTorch hook registration and management."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock
from typing import Any

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from tracetorch.collector import LayerTrace, TensorInfo, TensorStats, TraceRecord
from tracetorch.utils import compute_tensor_stats

# PyTorch's ``torch.cuda.Event`` is typed loosely in the bundled stubs; bind
# the constructor as a typed callable so call sites stay ignore-free.
_new_cuda_event: Callable[..., Any] = torch.cuda.Event

# Forward pass timing markers. Either a perf_counter float (CPU) or a
# torch.cuda.Event (CUDA); discriminated at record time.
TimingMark = float | Any


def _is_event(value: TimingMark) -> bool:
    """True when ``value`` is a CUDA event (vs a perf_counter float)."""
    return not isinstance(value, float)


class HookManager:
    """Manages PyTorch forward/backward hooks on a model.

    Repeated forwards within a single trace session are **aggregated** per
    module instance rather than appended as duplicates: latency min/max/mean
    and ``forward_count`` accumulate, the latest stats replace prior ones, and
    gradient observations merge into the same ``LayerTrace``.
    """

    def __init__(
        self,
        model: nn.Module,
        record: TraceRecord,
        *,
        capture_stats: bool = True,
    ) -> None:
        self._model = model
        self._record = record
        self._capture_stats = capture_stats
        self._hooks: list[RemovableHandle] = []
        # Per-module traces (keyed by id(module)) used both for aggregation
        # and for the backward hook to find the originating forward layer.
        self._layer_traces: dict[int, LayerTrace] = {}
        self._start_times: dict[int, TimingMark] = {}
        # Per-module GPU memory snapshot at forward/backward start, keyed by
        # ``id(module)``. Only populated when CUDA is available.
        self._mem_starts: dict[int, tuple[float, float]] = {}
        # Backward-start timing marks, one per module per backward pass.
        # Mirrors ``_start_times`` which tracks forward starts.
        self._bwd_start_times: dict[int, TimingMark] = {}
        self._active = False
        self._lock = Lock()
        # CUDA availability is static for the process; checked once.
        self._cuda_available = torch.cuda.is_available()

    @property
    def active(self) -> bool:
        return self._active

    def register(self) -> None:
        """Register forward + backward hooks on all nn.Module children."""
        for module_name, module in self._model.named_modules():
            if module is self._model and module_name == "":
                continue
            if not isinstance(module, nn.Module):
                continue

            self._hooks.append(module.register_forward_pre_hook(self._make_pre_hook()))
            self._hooks.append(module.register_forward_hook(self._make_forward_hook(module_name)))
            self._hooks.append(
                module.register_full_backward_hook(self._make_backward_hook(module_name))
            )
            # Backward pre-hook gives us a memory snapshot at the start of the
            # module's backward pass; not all PyTorch versions expose it, so
            # guard by attribute availability.
            pre_hook_fn = getattr(module, "register_full_backward_pre_hook", None)
            if callable(pre_hook_fn):
                self._hooks.append(pre_hook_fn(self._make_backward_pre_hook()))

    def unregister(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def reset(self) -> None:
        """Drop all per-session mutable state so the manager can be re-entered."""
        with self._lock:
            self._layer_traces.clear()
            self._start_times.clear()
            self._mem_starts.clear()
            self._bwd_start_times.clear()

    def enable(self) -> None:
        self._active = True

    def disable(self) -> None:
        self._active = False

    # ---------------------------------------------------------------- timing

    def _mark_start(self, mod: nn.Module) -> TimingMark:
        """Record a forward-start timing mark (CPU clock or CUDA event)."""
        if self._cuda_available and self._module_is_cuda(mod):
            start = _new_cuda_event(enable_timing=True)
            start.record()
            return start
        return time.perf_counter()

    @staticmethod
    def _mem_snapshot() -> tuple[float, float]:
        """Snapshot (allocated, reserved) bytes from the CUDA allocator.

        Caller must have verified CUDA is available; this method performs no
        availability check so it stays cheap on the CPU path which never calls
        it.
        """
        return (
            float(torch.cuda.memory_allocated()),
            float(torch.cuda.memory_reserved()),
        )

    def _record_mem_start(self, mod: nn.Module) -> None:
        """Stash the GPU memory snapshot at the start of a pass for ``mod``."""
        if self._cuda_available and self._module_is_cuda(mod):
            self._mem_starts[id(mod)] = self._mem_snapshot()

    def _apply_mem_delta(self, mod: nn.Module, trace: LayerTrace, *, backward: bool) -> None:
        """Compute and assign per-layer GPU memory deltas (forward or backward)."""
        if not self._cuda_available:
            return
        start = self._mem_starts.pop(id(mod), None)
        if start is None:
            return
        end_alloc, end_reserved = self._mem_snapshot()
        d_alloc = end_alloc - start[0]
        d_reserved = end_reserved - start[1]
        if backward:
            trace.bwd_mem_alloc_delta = d_alloc
            trace.bwd_mem_reserved_delta = d_reserved
        else:
            trace.fwd_mem_alloc_delta = d_alloc
            trace.fwd_mem_reserved_delta = d_reserved

    @staticmethod
    def _elapsed_ms(start: TimingMark, mod: nn.Module) -> float:
        """Resolve a start mark to elapsed milliseconds at forward end."""
        if _is_event(start):
            end = _new_cuda_event(enable_timing=True)
            end.record()
            # Synchronize so the events are populated; this is the single
            # host-side stall per layer per forward on CUDA.
            end.synchronize()
            event: Any = start
            return float(event.elapsed_time(end))
        # CPU path
        _ = mod
        return (time.perf_counter() - start) * 1000.0

    def _module_is_cuda(self, mod: nn.Module) -> bool:
        for p in mod.parameters(recurse=False):
            return p.is_cuda
        for b in mod.buffers(recurse=False):
            return b.is_cuda
        return False

    # ---------------------------------------------------------------- hooks

    def _make_pre_hook(self) -> Callable[..., Any]:
        def pre_hook(mod: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not self._active:
                return
            self._start_times[id(mod)] = self._mark_start(mod)
            self._record_mem_start(mod)

        return pre_hook

    def _make_forward_hook(self, module_name: str) -> Callable[..., Any]:
        def hook(
            mod: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if not self._active:
                return

            start = self._start_times.pop(id(mod), None)
            latency_ms = self._elapsed_ms(start, mod) if start is not None else 0.0

            with self._lock:
                existing = self._layer_traces.get(id(mod))
                if existing is not None:
                    self._update_layer(existing, inputs, output, latency_ms)
                    self._apply_mem_delta(mod, existing, backward=False)
                    return
                layer = self._trace_layer(module_name, mod, inputs, output, latency_ms)
                self._apply_mem_delta(mod, layer, backward=False)
                self._layer_traces[id(mod)] = layer
                self._record.layers.append(layer)

        return hook

    def _make_backward_pre_hook(self) -> Callable[..., Any]:
        def pre_hook(mod: nn.Module, _grad_in: tuple[Any, ...]) -> None:
            if not self._active:
                return
            # Record both the backward timing mark and a memory snapshot so
            # the post-hook can compute deltas symmetrically with the forward
            # path. ``_record_mem_start`` is a no-op on CPU.
            self._bwd_start_times[id(mod)] = self._mark_start(mod)
            self._record_mem_start(mod)

        return pre_hook

    def _make_backward_hook(self, module_name: str) -> Callable[..., Any]:
        def hook(
            mod: nn.Module,
            grad_in: tuple[Any, ...],
            grad_out: tuple[Any, ...],
        ) -> None:
            if not self._active:
                return

            with self._lock:
                layer = self._layer_traces.get(id(mod))
            if layer is None:
                return

            # Backward latency (mirrors the forward pre/post timing).
            bwd_start = self._bwd_start_times.pop(id(mod), None)
            bwd_latency_ms = self._elapsed_ms(bwd_start, mod) if bwd_start is not None else 0.0
            layer.record_backward_latency(bwd_latency_ms)

            self._apply_mem_delta(mod, layer, backward=True)

            # ``grad_in`` is the gradient arriving at the layer's INPUTS --
            # what the PREVIOUS layer (the one upstream in backward) handed
            # us. Useful for spotting broken gradient paths (a layer that
            # zeroes gradients is detectable here even when ``grad_out``
            # itself looks fine).
            in_norm_max: float | None = None
            in_mean_sum: float = 0.0
            in_count = 0
            in_has_nan = False
            for grad in grad_in:
                if not isinstance(grad, torch.Tensor):
                    continue
                if not grad.is_floating_point():
                    continue
                flat = grad.flatten()
                n = float(flat.norm().item())
                in_mean_sum += float(flat.mean().item())
                in_count += 1
                if in_norm_max is None or n > in_norm_max:
                    in_norm_max = n
                if torch.isnan(flat).any():
                    in_has_nan = True
            if in_count > 0 and in_norm_max is not None:
                layer.grad_in_norm = in_norm_max
                layer.grad_in_mean = in_mean_sum / in_count
                if in_has_nan:
                    layer.grad_in_has_nan = True

            # ``grad_out`` is the gradient flowing out of the layer's
            # outputs and back into its parameters -- this is what most users
            # associate with "the layer's gradient".
            for grad in grad_out:
                if not isinstance(grad, torch.Tensor):
                    continue
                if not grad.is_floating_point():
                    continue
                flat = grad.flatten()
                # Avoid materialising a full boolean mask: sum of abs handles
                # the common zero-grad case with one reduction and no allocation.
                norm_val = float(flat.norm().item())
                sum_abs = float(flat.abs().sum().item())
                layer.grad_norm = norm_val
                layer.grad_mean = float(flat.mean().item())
                if torch.isnan(flat).any():
                    layer.grad_has_nan = True
                if sum_abs == 0.0:
                    layer.has_zero_grad = True

        return hook

    # ---------------------------------------------------------------- trace

    def _trace_layer(
        self,
        module_name: str,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
        latency_ms: float,
    ) -> LayerTrace:
        depth = module_name.count(".") + 1 if module_name else 0
        param_count = sum(p.numel() for p in module.parameters(recurse=False))

        trace = LayerTrace(
            name=module_name.rsplit(".", 1)[-1] if module_name else module.__class__.__name__,
            module_type=type(module).__name__,
            full_name=module_name,
            depth=depth,
            params=param_count,
        )
        trace.record_latency(latency_ms)

        self._populate_tensors(trace, inputs, output)
        self._flag_anomalies(trace)
        return trace

    def _update_layer(
        self,
        trace: LayerTrace,
        inputs: tuple[Any, ...],
        output: Any,
        latency_ms: float,
    ) -> None:
        """Update an existing layer trace with a subsequent forward sample."""
        trace.record_latency(latency_ms)
        # Replace the latest input/output snapshots.
        trace.inputs.clear()
        trace.outputs.clear()
        self._populate_tensors(trace, inputs, output)
        self._flag_anomalies(trace)

    def _populate_tensors(
        self,
        trace: LayerTrace,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        for inp in self._iter_tensors(inputs):
            trace.inputs.append(self._tensor_info(inp))
        for out in self._iter_tensors(output):
            trace.outputs.append(self._tensor_info(out))

    def _tensor_info(self, tensor: torch.Tensor) -> TensorInfo:
        stats: TensorStats | None = None
        if self._capture_stats:
            stats_dict = compute_tensor_stats(tensor)
            stats = TensorStats(**stats_dict)
        return TensorInfo(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            stats=stats,
        )

    @staticmethod
    def _flag_anomalies(trace: LayerTrace) -> None:
        for info in trace.outputs:
            if info.stats is not None:
                if info.stats.nan_count > 0:
                    trace.has_nan = True
                if info.stats.inf_count > 0:
                    trace.has_inf = True

    @staticmethod
    def _iter_tensors(data: Any) -> list[torch.Tensor]:
        """Recursively extract tensors from nested structures."""
        tensors: list[torch.Tensor] = []
        if isinstance(data, torch.Tensor):
            tensors.append(data)
        elif isinstance(data, (tuple, list)):
            for item in data:
                tensors.extend(HookManager._iter_tensors(item))
        elif isinstance(data, dict):
            for value in data.values():
                tensors.extend(HookManager._iter_tensors(value))
        return tensors