"""Trace serialization and storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tracetorch.collector import TensorInfo, TraceRecord


def export_json(record: TraceRecord, path: str | Path) -> Path:
    """Export a TraceRecord to a JSON file.

    Args:
        record: The trace record to export.
        path: Destination file path. Parent directories are created automatically.

    Returns:
        The resolved Path of the written file.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = record.to_dict()
    with filepath.open("w", encoding="utf-8") as f:
        # ``default=str`` so non-JSON-native values in user-supplied metadata
        # (``torch.device``, ``dtype``, custom objects) stringify instead of
        # raising ``TypeError: Object of type X is not JSON serializable``.
        # Nested dataclasses are already serialised via ``to_dict()`` so this
        # only catches the rare escape hatch cases.
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    return filepath


def _tensor_info_from_dict(d: dict[str, Any]) -> TensorInfo:
    """Reconstruct a TensorInfo from a flat dict.

    The export format flattens stats fields into the top-level dict.
    """
    from tracetorch.collector import TensorStats

    stats_keys = {"mean", "std", "min", "max", "nan_count", "inf_count"}
    has_stats = any(k in d for k in stats_keys)

    stats = None
    if has_stats:
        stats = TensorStats(
            mean=d.get("mean"),
            std=d.get("std"),
            min=d.get("min"),
            max=d.get("max"),
            nan_count=d.get("nan_count", 0),
            inf_count=d.get("inf_count", 0),
        )

    return TensorInfo(
        shape=d["shape"],
        dtype=d["dtype"],
        device=d.get("device", ""),
        stats=stats,
    )


def load_json(path: str | Path) -> TraceRecord:
    """Load a TraceRecord from a JSON file.

    Args:
        path: Path to the JSON trace file.

    Returns:
        A reconstructed TraceRecord.
    """
    from tracetorch.collector import LayerTrace

    filepath = Path(path)
    with filepath.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    record = TraceRecord(metadata=data.get("metadata", {}))

    for layer_dict in data.get("layers", []):
        inputs = [_tensor_info_from_dict(t) for t in layer_dict.get("inputs", [])]
        outputs = [_tensor_info_from_dict(t) for t in layer_dict.get("outputs", [])]

        record.layers.append(
            LayerTrace(
                name=layer_dict["name"],
                module_type=layer_dict["type"],
                full_name=layer_dict.get("full_name", layer_dict["name"]),
                depth=layer_dict.get("depth", 0),
                params=layer_dict.get("params", 0),
                inputs=inputs,
                outputs=outputs,
                latency_ms=layer_dict.get("latency_ms", 0.0),
                latency_ms_min=layer_dict.get("latency_ms_min"),
                latency_ms_max=layer_dict.get("latency_ms_max"),
                latency_ms_mean=layer_dict.get("latency_ms_mean"),
                forward_count=layer_dict.get(
                    "forward_count", 1 if layer_dict.get("latency_ms", 0.0) else 0
                ),
                has_nan=layer_dict.get("has_nan", False),
                has_inf=layer_dict.get("has_inf", False),
                grad_norm=layer_dict.get("grad_norm"),
                grad_mean=layer_dict.get("grad_mean"),
                grad_has_nan=layer_dict.get("grad_has_nan", False),
                has_zero_grad=layer_dict.get("has_zero_grad", False),
                grad_in_norm=layer_dict.get("grad_in_norm"),
                grad_in_mean=layer_dict.get("grad_in_mean"),
                grad_in_has_nan=layer_dict.get("grad_in_has_nan", False),
                bwd_latency_ms=layer_dict.get("bwd_latency_ms", 0.0),
                bwd_latency_ms_min=layer_dict.get("bwd_latency_ms_min"),
                bwd_latency_ms_max=layer_dict.get("bwd_latency_ms_max"),
                bwd_latency_ms_mean=layer_dict.get("bwd_latency_ms_mean"),
                backward_count=layer_dict.get(
                    "backward_count", 1 if layer_dict.get("bwd_latency_ms", 0.0) else 0
                ),
                fwd_mem_alloc_delta=layer_dict.get("fwd_mem_alloc_delta"),
                fwd_mem_reserved_delta=layer_dict.get("fwd_mem_reserved_delta"),
                bwd_mem_alloc_delta=layer_dict.get("bwd_mem_alloc_delta"),
                bwd_mem_reserved_delta=layer_dict.get("bwd_mem_reserved_delta"),
            )
        )

    record.warnings = data.get("warnings", [])
    return record
