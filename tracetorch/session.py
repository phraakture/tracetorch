"""TraceSession — the main entry point for TraceTorch."""

from __future__ import annotations

import platform
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tracetorch.analyzer import Anomaly, Thresholds, detect_anomalies
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

    Live watch mode:
        Pass ``watch_path`` to have the session periodically flush its
        in-progress trace to disk during long training runs. Point
        ``tracetorch watch <path>`` at the same file for a live terminal
        view that re-renders on each write::

            session = TraceSession(model, watch_path="./trace.json",
                                   watch_interval_s=2.0)

            with session:
                for batch in dataloader:      # long training loop
                    ...

        The watch thread is a daemon and writes a best-effort snapshot — the
        exported JSON reflects the partially-captured state at flush time,
        without anomaly analysis (anomalies are computed once on ``__exit__``).

    Threading contract:
        Hooks fire from whatever thread runs the forward / backward pass.
        Per-session mutable state (``record.layers``, the hook manager's
        ``_layer_traces`` / ``_start_times`` / ``_mem_starts`` dicts) is
        guarded by an internal lock. The session itself is **not** safe to
        share across threads for simultaneous ``with`` blocks; a single
        session is meant to wrap a single training / inference loop on one
        thread. For parallel data loading or DataLoader workers the session
        lives on the main thread and observes the main-thread forward only.
        ``__enter__`` cannot be called twice without an intervening
        ``__exit__``; doing so raises ``RuntimeError``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        capture_stats: bool = True,
        model_name: str | None = None,
        thresholds: Thresholds | None = None,
        watch_path: str | Path | None = None,
        watch_interval_s: float = 1.0,
    ) -> None:
        """Initialize a TraceSession.

        Args:
            model: The PyTorch model to trace.
            capture_stats: Whether to compute activation statistics (mean, std, etc.).
            model_name: Optional name for the model in trace metadata.
            thresholds: Optional override of anomaly-detection thresholds. When
                ``None``, ``analyzer.DEFAULT_THRESHOLDS`` is used.
            watch_path: Optional path to periodically flush the in-progress
                trace to JSON during the session. Useful with
                ``tracetorch watch <path>`` for live observation of long
                training runs. The file is written at ``watch_interval_s``
                intervals and once more on ``__exit__``.
            watch_interval_s: Seconds between watch-mode flushes (default 1.0).
        """
        self._model = model
        self._capture_stats = capture_stats
        self._model_name = model_name or type(model).__name__
        self._thresholds = thresholds
        self._record = TraceRecord()
        self._hook_manager = HookManager(model, self._record, capture_stats=capture_stats)
        self._start_time: float = 0.0
        self._anomalies: list[Anomaly] = []
        # Re-entrancy guard: True between __enter__ and __exit__.
        self._entered: bool = False
        # Watch-mode state.
        self._watch_path: Path | None = Path(watch_path) if watch_path else None
        self._watch_interval_s: float = max(watch_interval_s, 0.1)
        self._watch_stop: threading.Event | None = None
        self._watch_thread: threading.Thread | None = None

    @property
    def record(self) -> TraceRecord:
        """The collected trace record."""
        return self._record

    @property
    def anomalies(self) -> list[Anomaly]:
        """Detected anomalies (available after exiting the context)."""
        return self._anomalies

    def __enter__(self) -> TraceSession:
        # Guard against accidental double-enter: re-registering hooks would
        # double-fire for each forwards, producing duplicate traces and
        # unremovable handles.
        if self._entered:
            raise RuntimeError(
                "TraceSession is already active -- cannot re-enter before __exit__."
            )
        self._entered = True

        # Reset all per-session state so a TraceSession can be safely re-entered
        # (or reused across multiple forwards) without stale layer traces,
        # stale backwards observations, or duplicate appends. Clear the record
        # in-place rather than replacing it so the HookManager (which holds a
        # reference to the same TraceRecord) keeps writing into the live one.
        self._record.layers.clear()
        self._record.warnings.clear()
        self._record.metadata = {}
        self._anomalies = []
        self._hook_manager.reset()
        self._start_time = time.perf_counter()
        self._record.metadata = self._build_metadata()
        self._hook_manager.register()
        self._hook_manager.enable()

        # Start the watch-mode flush thread if requested.
        if self._watch_path is not None:
            self._watch_stop = threading.Event()
            self._watch_thread = threading.Thread(
                target=self._watch_loop,
                name="tracetorch-watch",
                daemon=True,
            )
            self._watch_thread.start()

        return self

    def __exit__(self, *args: Any) -> None:
        self._entered = False
        self._hook_manager.disable()
        self._hook_manager.unregister()

        # Signal the watch thread to stop and wait for it to finish so the
        # final export below is the last write.
        if self._watch_stop is not None:
            self._watch_stop.set()
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=self._watch_interval_s + 1.0)
            self._watch_thread = None
            self._watch_stop = None

        # Anomaly analysis reads ``record.layers`` which may still be touched
        # by racing hook callbacks under unusual threading setups; the hook
        # manager's lock guards writes from hooks, and we disable hooks before
        # reading so no new appends can occur after this point.
        self._record.metadata["total_time_ms"] = (time.perf_counter() - self._start_time) * 1000
        # On CUDA, ensure any in-flight kernels have completed before we walk
        # the captured stats / latency so the recorded values are final.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._anomalies = detect_anomalies(self._record, thresholds=self._thresholds)
        self._record.warnings = [a.to_dict() for a in self._anomalies]

        # Final flush so the watch file reflects the complete, post-anomaly
        # state including warnings and total_time_ms.
        if self._watch_path is not None:
            export_json(self._record, self._watch_path)

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
        from tracetorch.utils import format_bytes, format_ms, format_params

        lines: list[str] = []
        lines.append(f"Model: {self._model_name}")
        lines.append(f"Layers traced: {len(self._record.layers)}")

        total_params = sum(layer.params for layer in self._record.layers)
        lines.append(f"Total parameters: {format_params(total_params)}")

        total_ms = self._record.metadata.get("total_time_ms", 0)
        lines.append(f"Total time: {format_ms(total_ms)}")

        total_forwards = max(
            (layer.forward_count for layer in self._record.layers),
            default=0,
        )
        if total_forwards > 0:
            lines.append(f"Forward passes: {total_forwards}")

        # Slowest layer by mean latency, where it's available.
        slowest = max(
            (layer for layer in self._record.layers),
            key=lambda layer: layer.latency_ms_mean or 0.0,
            default=None,
        )
        if slowest is not None and (slowest.latency_ms_mean or 0) > 0:
            lines.append(
                f"Slowest layer: {slowest.full_name} "
                f"({format_ms(slowest.latency_ms_mean or 0)} mean / "
                f"{format_ms(slowest.latency_ms_max or 0)} max over "
                f"{slowest.forward_count} fwd)"
            )

        # Slowest backward pass (only when one ran).
        bwd_layers = [
            layer
            for layer in self._record.layers
            if (layer.bwd_latency_ms_mean or 0) > 0
        ]
        if bwd_layers:
            slowest_bwd = max(
                bwd_layers, key=lambda layer: layer.bwd_latency_ms_mean or 0.0
            )
            lines.append(
                f"Slowest backward: {slowest_bwd.full_name} "
                f"({format_ms(slowest_bwd.bwd_latency_ms_mean or 0)} mean / "
                f"{format_ms(slowest_bwd.bwd_latency_ms_max or 0)} max over "
                f"{slowest_bwd.backward_count} bwd)"
            )

        # GPU memory: largest forward alloc delta surfaces the allocator-heavy
        # layer (often the OOM culprit). Only shown when CUDA traces ran.
        fwd_mem_layers = [
            layer
            for layer in self._record.layers
            if layer.fwd_mem_alloc_delta is not None
        ]
        if fwd_mem_layers:
            heaviest = max(
                fwd_mem_layers, key=lambda layer: layer.fwd_mem_alloc_delta or 0.0
            )
            if (heaviest.fwd_mem_alloc_delta or 0) > 0:
                lines.append(
                    f"Heaviest fwd mem: {heaviest.full_name} "
                    f"(+{format_bytes(heaviest.fwd_mem_alloc_delta)} alloc, "
                    f"+{format_bytes(heaviest.fwd_mem_reserved_delta)} reserved)"
                )
            bwd_mem_layers = [
                layer
                for layer in self._record.layers
                if layer.bwd_mem_alloc_delta is not None
            ]
            if bwd_mem_layers:
                heaviest_bwd = max(
                    bwd_mem_layers, key=lambda layer: layer.bwd_mem_alloc_delta or 0.0
                )
                if (heaviest_bwd.bwd_mem_alloc_delta or 0) > 0:
                    lines.append(
                        f"Heaviest bwd mem: {heaviest_bwd.full_name} "
                        f"(+{format_bytes(heaviest_bwd.bwd_mem_alloc_delta)} alloc, "
                        f"+{format_bytes(heaviest_bwd.bwd_mem_reserved_delta)} reserved)"
                    )

        if self._anomalies:
            lines.append(f"Anomalies detected: {len(self._anomalies)}")
            for a in self._anomalies:
                icon = "✗" if a.severity.value == "critical" else "⚠"
                lines.append(f"  {icon} [{a.severity.value}] {a.message}")

        return "\n".join(lines)

    def _watch_loop(self) -> None:
        """Background thread: periodically flush the partial record to disk.

        The snapshot is best-effort — ``record.layers`` may be mutated by
        hooks on the main thread while we serialise, but the export reads
        mostly scalar fields that are atomically assigned in CPython. The
        resulting JSON is a correct partial view at worst slightly stale,
        which is exactly what a live watch feature should display.
        """
        assert self._watch_stop is not None
        assert self._watch_path is not None
        while not self._watch_stop.wait(self._watch_interval_s):
            try:
                # Update total_time_ms so the watch view shows a live timer
                # rather than 0 until __exit__.
                self._record.metadata["total_time_ms"] = (
                    time.perf_counter() - self._start_time
                ) * 1000
                export_json(self._record, self._watch_path)
            except Exception:
                # Swallow errors in the watch thread so a training run never
                # crashes because the watch file couldn't be written.
                pass

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
