"""TraceSession — the main entry point for TraceTorch."""

from __future__ import annotations

import platform
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tracetorch.analyzer import Anomaly, detect_anomalies
from tracetorch.collector import TraceRecord
from tracetorch.hooks import HookManager
from tracetorch.storage import export_json


class TraceSession(AbstractContextManager["TraceSession"]):
    """Context manager that traces a PyTorch model during execution.

    Usage::

        model = MyModel()
        session = TraceSession(model)

        with session:
            output = model(input_tensor)

        session.export("./trace.json")
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        capture_stats: bool = True,
        model_name: str | None = None,
    ) -> None:
        """Initialize a TraceSession.

        Args:
            model: The PyTorch model to trace.
            capture_stats: Whether to compute activation statistics (mean, std, etc.).
            model_name: Optional name for the model in trace metadata.
        """
        self._model = model
        self._capture_stats = capture_stats
        self._model_name = model_name or type(model).__name__
        self._record = TraceRecord()
        self._hook_manager = HookManager(model, self._record)
        self._start_time: float = 0.0
        self._anomalies: list[Anomaly] = []

    @property
    def record(self) -> TraceRecord:
        """The collected trace record."""
        return self._record

    @property
    def anomalies(self) -> list[Anomaly]:
        """Detected anomalies (available after exiting the context)."""
        return self._anomalies

    def __enter__(self) -> TraceSession:
        self._start_time = time.perf_counter()
        self._record.metadata = self._build_metadata()
        self._hook_manager.register()
        self._hook_manager.enable()
        return self

    def __exit__(self, *args: Any) -> None:
        self._hook_manager.disable()
        self._hook_manager.unregister()
        self._record.metadata["total_time_ms"] = (time.perf_counter() - self._start_time) * 1000
        self._anomalies = detect_anomalies(self._record)
        self._record.warnings = [a.to_dict() for a in self._anomalies]

    def export(self, path: str | Path) -> Path:
        """Export the trace record to a JSON file.

        Args:
            path: Destination file path.

        Returns:
            The Path of the written file.
        """
        return export_json(self._record, path)

    def summary(self) -> str:
        """Return a human-readable summary of the trace."""
        from tracetorch.utils import format_ms, format_params

        lines: list[str] = []
        lines.append(f"Model: {self._model_name}")
        lines.append(f"Layers traced: {len(self._record.layers)}")

        total_params = sum(layer.params for layer in self._record.layers)
        lines.append(f"Total parameters: {format_params(total_params)}")

        total_ms = self._record.metadata.get("total_time_ms", 0)
        lines.append(f"Total time: {format_ms(total_ms)}")

        if self._anomalies:
            lines.append(f"Anomalies detected: {len(self._anomalies)}")
            for a in self._anomalies:
                icon = "✗" if a.severity.value == "critical" else "⚠"
                lines.append(f"  {icon} [{a.severity.value}] {a.message}")

        return "\n".join(lines)

    def _build_metadata(self) -> dict[str, Any]:
        params = list(self._model.parameters())
        active_device = str(params[0].device) if params else "cpu"
        return {
            "model_name": self._model_name,
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda or "N/A",
            "active_device": active_device,
        }
