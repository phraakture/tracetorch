"""Data structures for trace records and layer information."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TensorStats:
    """Statistics captured from a tensor during execution."""

    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    nan_count: int = 0
    inf_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


@dataclass
class TensorInfo:
    """Shape, dtype, and device information for a tensor."""

    shape: list[int]
    dtype: str
    device: str = ""
    stats: TensorStats | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self.device,
        }
        if self.stats is not None:
            d.update(self.stats.to_dict())
        return d


@dataclass
class LayerTrace:
    """Complete trace data for a single nn.Module layer."""

    name: str
    module_type: str
    full_name: str
    depth: int
    params: int
    inputs: list[TensorInfo] = field(default_factory=list)
    outputs: list[TensorInfo] = field(default_factory=list)
    # Most recent forward latency (milliseconds). Kept for backward compat.
    latency_ms: float = 0.0
    # Aggregated latency across repeated forwards (milliseconds).
    latency_ms_min: float | None = None
    latency_ms_max: float | None = None
    latency_ms_mean: float | None = None
    forward_count: int = 0
    has_nan: bool = False
    has_inf: bool = False
    # Gradient fields (populated during backward pass). ``grad_*`` reflect the
    # gradient flowing INTO the layer's outputs (``grad_out``); ``grad_in_*``
    # capture the gradient flowing INTO the layer's inputs, useful for
    # detecting broken gradient paths (e.g. a layer that drops gradients).
    grad_norm: float | None = None
    grad_mean: float | None = None
    grad_has_nan: bool = False
    has_zero_grad: bool = False
    grad_in_norm: float | None = None
    grad_in_mean: float | None = None
    grad_in_has_nan: bool = False
    # Backward latency (milliseconds). Mirrors the forward latency fields so
    # backward cost can be profiled symmetrically. Populated only when a
    # backward pass runs through the layer.
    bwd_latency_ms: float = 0.0
    bwd_latency_ms_min: float | None = None
    bwd_latency_ms_max: float | None = None
    bwd_latency_ms_mean: float | None = None
    backward_count: int = 0
    # Per-layer GPU memory deltas in bytes (CUDA only; None on CPU).
    # Forward: change in `torch.cuda.memory_allocated()` from pre- to
    # post-forward hook. Reserved-memory delta uses `memory_reserved()`.
    fwd_mem_alloc_delta: float | None = None
    fwd_mem_reserved_delta: float | None = None
    # Backward deltas (populated if a backward pass runs).
    bwd_mem_alloc_delta: float | None = None
    bwd_mem_reserved_delta: float | None = None

    def record_latency(self, latency_ms: float) -> None:
        """Accumulate a forward latency sample into the aggregate fields."""
        self.latency_ms = latency_ms
        self.forward_count += 1
        if self.latency_ms_min is None or latency_ms < self.latency_ms_min:
            self.latency_ms_min = latency_ms
        if self.latency_ms_max is None or latency_ms > self.latency_ms_max:
            self.latency_ms_max = latency_ms
        prev_sum = (self.latency_ms_mean or 0.0) * (self.forward_count - 1)
        self.latency_ms_mean = (prev_sum + latency_ms) / self.forward_count

    def record_backward_latency(self, latency_ms: float) -> None:
        """Accumulate a backward latency sample into the aggregate fields."""
        self.bwd_latency_ms = latency_ms
        self.backward_count += 1
        if self.bwd_latency_ms_min is None or latency_ms < self.bwd_latency_ms_min:
            self.bwd_latency_ms_min = latency_ms
        if self.bwd_latency_ms_max is None or latency_ms > self.bwd_latency_ms_max:
            self.bwd_latency_ms_max = latency_ms
        prev_sum = (self.bwd_latency_ms_mean or 0.0) * (self.backward_count - 1)
        self.bwd_latency_ms_mean = (prev_sum + latency_ms) / self.backward_count

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.module_type,
            "full_name": self.full_name,
            "depth": self.depth,
            "params": self.params,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "latency_ms": self.latency_ms,
            "latency_ms_min": self.latency_ms_min,
            "latency_ms_max": self.latency_ms_max,
            "latency_ms_mean": self.latency_ms_mean,
            "forward_count": self.forward_count,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "grad_norm": self.grad_norm,
            "grad_mean": self.grad_mean,
            "grad_has_nan": self.grad_has_nan,
            "has_zero_grad": self.has_zero_grad,
            "grad_in_norm": self.grad_in_norm,
            "grad_in_mean": self.grad_in_mean,
            "grad_in_has_nan": self.grad_in_has_nan,
            "bwd_latency_ms": self.bwd_latency_ms,
            "bwd_latency_ms_min": self.bwd_latency_ms_min,
            "bwd_latency_ms_max": self.bwd_latency_ms_max,
            "bwd_latency_ms_mean": self.bwd_latency_ms_mean,
            "backward_count": self.backward_count,
            "fwd_mem_alloc_delta": self.fwd_mem_alloc_delta,
            "fwd_mem_reserved_delta": self.fwd_mem_reserved_delta,
            "bwd_mem_alloc_delta": self.bwd_mem_alloc_delta,
            "bwd_mem_reserved_delta": self.bwd_mem_reserved_delta,
        }
        return d


@dataclass
class TraceRecord:
    """A complete trace record containing metadata, layer traces, and anomalies."""

    metadata: dict[str, Any] = field(default_factory=dict)
    layers: list[LayerTrace] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "layers": [layer.to_dict() for layer in self.layers],
            "warnings": self.warnings,
        }
