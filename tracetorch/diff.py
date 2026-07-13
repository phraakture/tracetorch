"""Diff two trace records for CI / regression use.

Given two :class:`TraceRecord` instances (baseline and current), produce a
structured comparison highlighting:

* layers added / removed (by ``full_name``)
* latency regressions (mean latency relative delta, with a configurable
  threshold expressed as a fraction, e.g. ``0.1`` for 10%)
* new or resolved anomalies
* gradient-norm deltas
* activation-std deltas

The output is a :class:`TraceDiff` dataclass with ``to_dict()`` for JSON
serialization, plus a small ``Severity`` classifier so the CLI can render it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from tracetorch.collector import LayerTrace, TraceRecord


class DiffSeverity(StrEnum):
    """Severity bucket for a single diff entry."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DiffEntry:
    """A single observed change between two traces."""

    severity: DiffSeverity
    kind: str
    layer: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "severity": self.severity.value,
            "kind": self.kind,
            "layer": self.layer,
            "message": self.message,
        }
        if self.details is not None:
            d["details"] = self.details
        return d


@dataclass
class TraceDiff:
    """Structured diff of two :class:`TraceRecord` objects."""

    baseline_model: str
    current_model: str
    entries: list[DiffEntry] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        """True if any critical or warning entry is present."""
        return any(
            e.severity in (DiffSeverity.CRITICAL, DiffSeverity.WARNING) for e in self.entries
        )

    @property
    def critical_count(self) -> int:
        return sum(1 for e in self.entries if e.severity == DiffSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for e in self.entries if e.severity == DiffSeverity.WARNING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_model": self.baseline_model,
            "current_model": self.current_model,
            "has_regressions": self.has_regressions,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "entries": [e.to_dict() for e in self.entries],
        }


def diff_records(
    baseline: TraceRecord,
    current: TraceRecord,
    *,
    latency_threshold: float = 0.1,
    grad_norm_threshold: float = 0.25,
    std_threshold: float = 0.25,
) -> TraceDiff:
    """Compare two trace records and return a structured :class:`TraceDiff`.

    Thresholds are expressed as **relative** deltas (``0.1`` = 10%). A latency
    regression is flagged when the current mean latency exceeds baseline by
    more than ``latency_threshold`` relative. An improvement below the
    negative threshold is reported as INFO so users can spot accidental
    speedups too.

    Args:
        baseline: The reference trace (e.g. main branch / last green build).
        current: The trace under review.
        latency_threshold: Relative delta that triggers a latency warning.
        grad_norm_threshold: Relative delta that triggers a grad-norm warning.
        std_threshold: Relative delta that triggers an activation-std warning.
    """
    base_layers = {layer.full_name: layer for layer in baseline.layers}
    curr_layers = {layer.full_name: layer for layer in current.layers}

    base_model = str(baseline.metadata.get("model_name", "unknown"))
    curr_model = str(current.metadata.get("model_name", "unknown"))
    diff = TraceDiff(baseline_model=base_model, current_model=curr_model)

    base_warns = {(w.get("type"), w.get("layer")) for w in baseline.warnings}
    curr_warns = {(w.get("type"), w.get("layer")) for w in current.warnings}

    # Added / removed layers.
    for name in curr_layers.keys() - base_layers.keys():
        diff.entries.append(
            DiffEntry(
                severity=DiffSeverity.WARNING,
                kind="layer_added",
                layer=name,
                message=f"Layer added in current trace: {name}",
                details={"type": curr_layers[name].module_type},
            )
        )
    for name in base_layers.keys() - curr_layers.keys():
        diff.entries.append(
            DiffEntry(
                severity=DiffSeverity.WARNING,
                kind="layer_removed",
                layer=name,
                message=f"Layer removed in current trace: {name}",
                details={"type": base_layers[name].module_type},
            )
        )

    # Per-layer comparisons.
    for name, base in base_layers.items():
        curr = curr_layers.get(name)
        if curr is None:
            continue
        _diff_latency(diff, name, base, curr, latency_threshold)
        _diff_grad_norm(diff, name, base, curr, grad_norm_threshold)
        _diff_activation_std(diff, name, base, curr, std_threshold)
        _diff_shapes(diff, name, base, curr)

    # New / resolved anomalies.
    for key in curr_warns - base_warns:
        w = _find_warning(current, key)
        diff.entries.append(
            DiffEntry(
                severity=DiffSeverity.CRITICAL
                if (w.get("severity") == "critical")
                else DiffSeverity.WARNING,
                kind="new_anomaly",
                layer=str(key[1] or ""),
                message=f"New anomaly: {w.get('message', key[0])}",
                details={"type": key[0], "severity": w.get("severity")},
            )
        )
    for key in base_warns - curr_warns:
        w = _find_warning(baseline, key)
        diff.entries.append(
            DiffEntry(
                severity=DiffSeverity.INFO,
                kind="resolved_anomaly",
                layer=str(key[1] or ""),
                message=f"Resolved anomaly: {w.get('message', key[0])}",
                details={"type": key[0]},
            )
        )

    return diff


def _diff_latency(
    diff: TraceDiff,
    name: str,
    base: LayerTrace,
    curr: LayerTrace,
    threshold: float,
) -> None:
    base_lat = base.latency_ms_mean if base.latency_ms_mean is not None else base.latency_ms
    curr_lat = curr.latency_ms_mean if curr.latency_ms_mean is not None else curr.latency_ms
    if base_lat <= 0 or curr_lat <= 0:
        return
    rel = (curr_lat - base_lat) / base_lat
    if rel > threshold:
        diff.entries.append(
            DiffEntry(
                severity=DiffSeverity.WARNING,
                kind="latency_regression",
                layer=name,
                message=(
                    f"Latency regression: {base_lat:.3f}ms -> {curr_lat:.3f}ms (+{rel * 100:.1f}%)"
                ),
                details={
                    "baseline_ms": base_lat,
                    "current_ms": curr_lat,
                    "relative": rel,
                    "baseline_count": base.forward_count,
                    "current_count": curr.forward_count,
                },
            )
        )
    elif rel < -threshold:
        diff.entries.append(
            DiffEntry(
                severity=DiffSeverity.INFO,
                kind="latency_improvement",
                layer=name,
                message=(
                    f"Latency improved: {base_lat:.3f}ms -> {curr_lat:.3f}ms ({rel * 100:.1f}%)"
                ),
                details={
                    "baseline_ms": base_lat,
                    "current_ms": curr_lat,
                    "relative": rel,
                },
            )
        )


def _diff_grad_norm(
    diff: TraceDiff,
    name: str,
    base: LayerTrace,
    curr: LayerTrace,
    threshold: float,
) -> None:
    if base.grad_norm is None or curr.grad_norm is None:
        return
    if base.grad_norm == 0:
        return
    rel = (curr.grad_norm - base.grad_norm) / abs(base.grad_norm)
    if abs(rel) > threshold:
        severity = DiffSeverity.WARNING if curr.grad_norm > base.grad_norm else DiffSeverity.INFO
        diff.entries.append(
            DiffEntry(
                severity=severity,
                kind="grad_norm_change",
                layer=name,
                message=(
                    f"Gradient norm changed: {base.grad_norm:.4f} -> "
                    f"{curr.grad_norm:.4f} ({rel * 100:+.1f}%)"
                ),
                details={
                    "baseline": base.grad_norm,
                    "current": curr.grad_norm,
                    "relative": rel,
                },
            )
        )


def _diff_activation_std(
    diff: TraceDiff,
    name: str,
    base: LayerTrace,
    curr: LayerTrace,
    threshold: float,
) -> None:
    base_std = _first_std(base)
    curr_std = _first_std(curr)
    if base_std is None or curr_std is None or base_std == 0:
        return
    rel = (curr_std - base_std) / base_std
    if abs(rel) > threshold:
        severity = DiffSeverity.WARNING if curr_std > base_std else DiffSeverity.INFO
        diff.entries.append(
            DiffEntry(
                severity=severity,
                kind="activation_std_change",
                layer=name,
                message=(
                    f"Output std changed: {base_std:.4f} -> {curr_std:.4f} ({rel * 100:+.1f}%)"
                ),
                details={
                    "baseline": base_std,
                    "current": curr_std,
                    "relative": rel,
                },
            )
        )


def _diff_shapes(
    diff: TraceDiff,
    name: str,
    base: LayerTrace,
    curr: LayerTrace,
) -> None:
    base_shapes = [tuple(o.shape) for o in base.outputs]
    curr_shapes = [tuple(o.shape) for o in curr.outputs]
    if base_shapes != curr_shapes:
        diff.entries.append(
            DiffEntry(
                severity=DiffSeverity.CRITICAL,
                kind="shape_change",
                layer=name,
                message=f"Output shape changed: {base_shapes} -> {curr_shapes}",
                details={"baseline": base_shapes, "current": curr_shapes},
            )
        )


def _find_warning(record: TraceRecord, key: tuple[Any, Any]) -> dict[str, Any]:
    for w in record.warnings:
        if (w.get("type"), w.get("layer")) == key:
            return w
    return {}


def _first_std(layer: LayerTrace) -> float | None:
    for info in layer.outputs:
        if info.stats is not None and info.stats.std is not None:
            return info.stats.std
    return None
