"""Anomaly detection for trace data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tracetorch.collector import LayerTrace, TensorInfo, TraceRecord


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


@dataclass(frozen=True)
class Thresholds:
    """Configurable thresholds for anomaly detection.

    All values are sensible defaults for typical float32 training. Override
    them via ``TraceSession(model, thresholds=Thresholds(dead_layer_std=1e-4,
    ...))`` to tune sensitivity for unusual models (e.g. quantized or
    sparse activations).
    """

    # ``dead_layer``: layer flagged when ``|mean| < dead_layer_mean`` and
    # ``std < dead_layer_std``. Defaults are tighter than the original 1e-8 in
    # name only -- the std threshold is raised to 1e-6 to avoid false positives
    # on layers that legitimately produce very small activations (e.g. an
    # L2-normalized projection or a saturated sigmoid at init).
    dead_layer_mean: float = 1e-6
    dead_layer_std: float = 1e-6

    # ``exploding_variance``: output std above this absolute value triggers
    # the warning. ``high_variance`` is the info-level band between
    # ``high_variance_min`` and ``exploding_variance_std``.
    exploding_variance_std: float = 100.0
    high_variance_min: float = 10.0

    # Per-layer input->output std ratio above which a variance spike is
    # flagged.
    variance_spike_ratio: float = 10.0


# Module-level default singleton. Use ``Thresholds()`` to override.
DEFAULT_THRESHOLDS = Thresholds()


def detect_anomalies(
    record: TraceRecord,
    thresholds: Thresholds | None = None,
) -> list[Anomaly]:
    """Analyze a trace record and return detected anomalies.

    Args:
        record: The trace record to inspect.
        thresholds: Optional override of anomaly-detection thresholds. When
            ``None``, ``DEFAULT_THRESHOLDS`` is used.
    """
    th = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    anomalies: list[Anomaly] = []

    for layer in record.layers:
        _check_nan(layer, anomalies)
        _check_inf(layer, anomalies)
        _check_dead_layer(layer, anomalies, th)
        _check_exploding_variance(layer, anomalies, th)
        _check_high_variance(layer, anomalies, th)
        _check_empty_output(layer, anomalies)
        _check_zero_gradient(layer, anomalies)
        _check_nan_gradient(layer, anomalies)

    # Cross-layer / per-layer checks
    _check_variance_spikes(record.layers, anomalies, th)

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


def _check_dead_layer(
    layer: LayerTrace,
    anomalies: list[Anomaly],
    th: Thresholds,
) -> None:
    """Detect layers whose output is all zeros or near-zero."""
    for info in layer.outputs:
        if info.stats is None:
            continue
        if info.stats.mean is None:
            continue
        if abs(info.stats.mean) < th.dead_layer_mean and (
            info.stats.std is not None and info.stats.std < th.dead_layer_std
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


def _check_exploding_variance(
    layer: LayerTrace,
    anomalies: list[Anomaly],
    th: Thresholds,
) -> None:
    """Detect layers with extremely high output variance."""
    for info in layer.outputs:
        if info.stats is None or info.stats.std is None:
            continue
        if info.stats.std > th.exploding_variance_std:
            anomalies.append(
                Anomaly(
                    type=AnomalyType.EXPLODING_VARIANCE,
                    severity=AnomalySeverity.WARNING,
                    layer=layer.full_name,
                    message=f"Activation variance exploded: std={info.stats.std:.2f}",
                    details={"std": info.stats.std},
                )
            )


def _check_high_variance(
    layer: LayerTrace,
    anomalies: list[Anomaly],
    th: Thresholds,
) -> None:
    """Detect layers with unusually high output variance (but not exploding)."""
    for info in layer.outputs:
        if info.stats is None or info.stats.std is None:
            continue
        if th.high_variance_min < info.stats.std <= th.exploding_variance_std:
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
    layers: list[LayerTrace],
    anomalies: list[Anomaly],
    th: Thresholds,
) -> None:
    """Per-layer input-to-output variance spike detection.

    A layer is flagged when its output std exceeds its input std by more than
    ``th.variance_spike_ratio``x. This is **order-independent**: each layer is
    evaluated against its own captured inputs, so branched models,
    control-flow models, or shared submodules no longer produce false
    positives from adjacent-but-unrelated layers that happened to run
    earlier.
    """
    for layer in layers:
        in_std = _first_std(layer.inputs)
        out_std = _first_std(layer.outputs)
        if in_std is None or out_std is None or in_std <= 0:
            continue
        ratio = out_std / in_std
        if ratio > th.variance_spike_ratio:
            anomalies.append(
                Anomaly(
                    type=AnomalyType.EXPLODING_VARIANCE,
                    severity=AnomalySeverity.WARNING,
                    layer=layer.full_name,
                    message=(
                        f"Variance spike: std increased from {in_std:.2f} "
                        f"to {out_std:.2f} (x{ratio:.1f})"
                    ),
                    details={
                        "input_std": in_std,
                        "output_std": out_std,
                        "ratio": ratio,
                    },
                )
            )


def _first_std(infos: list[TensorInfo]) -> float | None:
    """Return the first non-None std from a list of TensorInfo, or None."""
    for info in infos:
        if info.stats is not None and info.stats.std is not None:
            return float(info.stats.std)
    return None
