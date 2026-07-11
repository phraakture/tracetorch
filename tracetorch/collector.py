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
    latency_ms: float = 0.0
    has_nan: bool = False
    has_inf: bool = False
    # Gradient fields (populated during backward pass)
    grad_norm: float | None = None
    grad_mean: float | None = None
    grad_has_nan: bool = False
    has_zero_grad: bool = False

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
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "grad_norm": self.grad_norm,
            "grad_mean": self.grad_mean,
            "grad_has_nan": self.grad_has_nan,
            "has_zero_grad": self.has_zero_grad,
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
