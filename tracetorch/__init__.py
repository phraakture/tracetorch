"""TraceTorch — Chrome DevTools for neural networks.

A production-grade Python library that gives ML engineers an interactive,
forensic view of what happens inside PyTorch models during execution.
"""

from __future__ import annotations

from tracetorch.analyzer import Anomaly, AnomalySeverity, AnomalyType
from tracetorch.collector import LayerTrace, TensorStats, TraceRecord
from tracetorch.session import TraceSession

__all__ = [
    "Anomaly",
    "AnomalySeverity",
    "AnomalyType",
    "LayerTrace",
    "TensorStats",
    "TraceRecord",
    "TraceSession",
]
