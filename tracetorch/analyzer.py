"""Anomaly detection for trace data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tracetorch.collector import LayerTrace, TraceRecord


class AnomalyType(StrEnum):
    """Types of anomalies that can be detected."""

    NAN_ACTIVATION = "nan_activation"
    INF_ACTIVATION = "inf_activation"
    EXPLODING_VARIANCE = "exploding_variance"
    DEAD_LAYER = "dead_layer"
    HIGH_VARIANCE = "high_variance"
    EMPTY_OUTPUT = "empty_output"
    ZERO_GRADIENT = "zero_gradient"
    NAN_GRADIENT = "nan_gradient"


class AnomalySeverity(StrEnum):
    """Severity levels for anomalies."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Anomaly:
    """A detected anomaly in the model trace."""

    type: AnomalyType
    severity: AnomalySeverity
    layer: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.type.value,
            "severity": self.severity.value,
            "layer": self.layer,
            "message": self.message,
        }
        if self.details is not None:
            d["details"] = self.details
        return d


def detect_anomalies(record: TraceRecord) -> list[Anomaly]:
    """Analyze a trace record and return detected anomalies."""
    anomalies: list[Anomaly] = []

    for layer in record.layers:
        _check_nan(layer, anomalies)
        _check_inf(layer, anomalies)
        _check_dead_layer(layer, anomalies)
        _check_exploding_variance(layer, anomalies)
        _check_high_variance(layer, anomalies)
        _check_empty_output(layer, anomalies)
        _check_zero_gradient(layer, anomalies)
        _check_nan_gradient(layer, anomalies)

    # Cross-layer checks
    _check_variance_spikes(record.layers, anomalies)

    return anomalies


def _check_nan(layer: LayerTrace, anomalies: list[Anomaly]) -> None:
    if not layer.has_nan:
        return
    anomalies.append(
        Anomaly(
            type=AnomalyType.NAN_ACTIVATION,
            severity=AnomalySeverity.CRITICAL,
            layer=layer.full_name,
            message=f"NaN detected in output of {layer.module_type}",
            details={"module_type": layer.module_type},
        )
    )


def _check_inf(layer: LayerTrace, anomalies: list[Anomaly]) -> None:
    if not layer.has_inf:
        return
    anomalies.append(
        Anomaly(
            type=AnomalyType.INF_ACTIVATION,
            severity=AnomalySeverity.CRITICAL,
            layer=layer.full_name,
            message=f"Infinite values detected in output of {layer.module_type}",
            details={"module_type": layer.module_type},
        )
    )


def _check_dead_layer(layer: LayerTrace, anomalies: list[Anomaly]) -> None:
    """Detect layers whose output is all zeros or near-zero."""
    for info in layer.outputs:
        if info.stats is None:
            continue
        if info.stats.mean is None:
            continue
        if (
            abs(info.stats.mean) < 1e-8
            and info.stats.std is not None
            and info.stats.std < 1e-8
        ):
            anomalies.append(
                Anomaly(
                    type=AnomalyType.DEAD_LAYER,
                    severity=AnomalySeverity.WARNING,
                    layer=layer.full_name,
                    message=f"Dead layer detected: {layer.module_type} output is near-zero",
                    details={
                        "mean": info.stats.mean,
                        "std": info.stats.std,
                        "shape": info.shape,
                    },
                )
            )


def _check_exploding_variance(layer: LayerTrace, anomalies: list[Anomaly]) -> None:
    """Detect layers with extremely high output variance."""
    for info in layer.outputs:
        if info.stats is None or info.stats.std is None:
            continue
        if info.stats.std > 100.0:
            anomalies.append(
                Anomaly(
                    type=AnomalyType.EXPLODING_VARIANCE,
                    severity=AnomalySeverity.WARNING,
                    layer=layer.full_name,
                    message=f"Activation variance exploded: std={info.stats.std:.2f}",
                    details={"std": info.stats.std},
                )
            )


def _check_high_variance(layer: LayerTrace, anomalies: list[Anomaly]) -> None:
    """Detect layers with unusually high output variance (but not exploding)."""
    for info in layer.outputs:
        if info.stats is None or info.stats.std is None:
            continue
        if 10.0 < info.stats.std <= 100.0:
            anomalies.append(
                Anomaly(
                    type=AnomalyType.HIGH_VARIANCE,
                    severity=AnomalySeverity.INFO,
                    layer=layer.full_name,
                    message=f"High activation variance: std={info.stats.std:.2f}",
                    details={"std": info.stats.std},
                )
            )


def _check_empty_output(layer: LayerTrace, anomalies: list[Anomaly]) -> None:
    """Detect layers that produced no output tensors."""
    if not layer.outputs:
        anomalies.append(
            Anomaly(
                type=AnomalyType.EMPTY_OUTPUT,
                severity=AnomalySeverity.WARNING,
                layer=layer.full_name,
                message=f"No output captured from {layer.module_type}",
            )
        )


def _check_zero_gradient(layer: LayerTrace, anomalies: list[Anomaly]) -> None:
    """Detect layers with all-zero gradients."""
    if not layer.has_zero_grad:
        return
    anomalies.append(
        Anomaly(
            type=AnomalyType.ZERO_GRADIENT,
            severity=AnomalySeverity.WARNING,
            layer=layer.full_name,
            message=f"Zero gradient detected in {layer.module_type} — may indicate dead pathway",
            details={"grad_norm": layer.grad_norm},
        )
    )


def _check_nan_gradient(layer: LayerTrace, anomalies: list[Anomaly]) -> None:
    """Detect layers with NaN gradients."""
    if not layer.grad_has_nan:
        return
    anomalies.append(
        Anomaly(
            type=AnomalyType.NAN_GRADIENT,
            severity=AnomalySeverity.CRITICAL,
            layer=layer.full_name,
            message=f"NaN gradient detected in {layer.module_type}",
            details={"grad_norm": layer.grad_norm},
        )
    )


def _check_variance_spikes(
    layers: list[LayerTrace], anomalies: list[Anomaly]
) -> None:
    """Compare variance across sequential layers to detect sudden spikes."""
    prev_std: float | None = None
    prev_name: str = ""

    for layer in layers:
        for info in layer.outputs:
            if info.stats is None or info.stats.std is None:
                continue
            if prev_std is not None and prev_std > 0:
                ratio = info.stats.std / prev_std
                if ratio > 10.0:
                    anomalies.append(
                        Anomaly(
                            type=AnomalyType.EXPLODING_VARIANCE,
                            severity=AnomalySeverity.WARNING,
                            layer=layer.full_name,
                            message=(
                                f"Variance spike: std increased from {prev_std:.2f} "
                                f"to {info.stats.std:.2f} (×{ratio:.1f})"
                            ),
                            details={
                                "previous_layer": prev_name,
                                "previous_std": prev_std,
                                "current_std": info.stats.std,
                                "ratio": ratio,
                            },
                        )
                    )
            prev_std = info.stats.std
            prev_name = layer.full_name
