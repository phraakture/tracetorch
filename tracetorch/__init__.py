"""TraceTorch — A forensic trace console for PyTorch models.

A production-grade Python library that gives ML engineers an interactive,
forensic view of what happens inside PyTorch models during execution.
"""

from __future__ import annotations

from tracetorch.analyzer import Anomaly, AnomalySeverity, AnomalyType, Thresholds
from tracetorch.collector import LayerTrace, TensorStats, TraceRecord
from tracetorch.diff import DiffSeverity, TraceDiff, diff_records
from tracetorch.session import TraceSession

__all__ = [
    "Anomaly",
    "AnomalySeverity",
    "AnomalyType",
    "DiffSeverity",
    "LayerTrace",
    "TensorStats",
    "Thresholds",
    "TraceDiff",
    "TraceRecord",
    "TraceSession",
    "diff_records",
]
