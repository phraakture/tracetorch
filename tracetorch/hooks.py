"""PyTorch hook registration and management."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from tracetorch.collector import LayerTrace, TensorInfo, TensorStats, TraceRecord
from tracetorch.utils import compute_tensor_stats


class HookManager:
    """Manages PyTorch forward/backward hooks on a model."""

    def __init__(self, model: nn.Module, record: TraceRecord) -> None:
        self._model = model
        self._record = record
        self._hooks: list[RemovableHandle] = []
        self._layer_traces: dict[int, LayerTrace] = {}
        self._start_times: dict[int, float] = {}
        self._active = False

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

            self._hooks.append(
                module.register_forward_pre_hook(self._make_pre_hook())
            )
            self._hooks.append(
                module.register_forward_hook(self._make_forward_hook(module_name))
            )
            self._hooks.append(
                module.register_full_backward_hook(self._make_backward_hook(module_name))
            )

    def unregister(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def enable(self) -> None:
        self._active = True

    def disable(self) -> None:
        self._active = False

    def _make_pre_hook(self) -> Callable[..., Any]:
        def pre_hook(mod: nn.Module, inputs: tuple[Any, ...]) -> None:
            if self._active:
                self._start_times[id(mod)] = time.perf_counter()

        return pre_hook

    def _make_forward_hook(
        self, module_name: str
    ) -> Callable[..., Any]:
        def hook(
            mod: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if not self._active:
                return

            t_start = self._start_times.pop(id(mod), time.perf_counter())
            layer = self._trace_layer(module_name, mod, inputs, output, t_start)
            self._layer_traces[id(mod)] = layer
            self._record.layers.append(layer)

        return hook

    def _make_backward_hook(
        self, module_name: str
    ) -> Callable[..., Any]:
        def hook(
            mod: nn.Module,
            grad_in: tuple[Any, ...],
            grad_out: tuple[Any, ...],
        ) -> None:
            if not self._active:
                return

            layer = self._layer_traces.get(id(mod))
            if layer is None:
                return

            for grad in grad_out:
                if not isinstance(grad, torch.Tensor):
                    continue
                if not grad.is_floating_point():
                    continue
                flat = grad.flatten().float()
                layer.grad_norm = float(flat.norm().item())
                layer.grad_mean = float(flat.mean().item())
                if torch.isnan(flat).any():
                    layer.grad_has_nan = True
                if torch.all(flat == 0):
                    layer.has_zero_grad = True

        return hook

    def _trace_layer(
        self,
        module_name: str,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
        t_start: float,
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

        # Capture input stats
        for inp in self._iter_tensors(inputs):
            stats_dict = compute_tensor_stats(inp)
            trace.inputs.append(
                TensorInfo(
                    shape=list(inp.shape),
                    dtype=str(inp.dtype),
                    device=str(inp.device),
                    stats=TensorStats(**stats_dict),
                )
            )

        # Capture output stats
        for out in self._iter_tensors(output):
            stats_dict = compute_tensor_stats(out)
            trace.outputs.append(
                TensorInfo(
                    shape=list(out.shape),
                    dtype=str(out.dtype),
                    device=str(out.device),
                    stats=TensorStats(**stats_dict),
                )
            )

        # Latency from pre-hook start to post-hook end
        trace.latency_ms = (time.perf_counter() - t_start) * 1000

        # Check for anomalies at capture time
        for info in trace.outputs:
            if info.stats is not None:
                if info.stats.nan_count > 0:
                    trace.has_nan = True
                if info.stats.inf_count > 0:
                    trace.has_inf = True

        return trace

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
