"""Tests for TraceTorch core functionality."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tracetorch import TraceSession
from tracetorch.analyzer import AnomalyType, Thresholds, detect_anomalies
from tracetorch.collector import LayerTrace, TensorInfo, TensorStats, TraceRecord
from tracetorch.diff import DiffSeverity, diff_records
from tracetorch.hooks import HookManager
from tracetorch.storage import export_json, load_json
from tracetorch.utils import (
    compute_tensor_stats,
    format_bytes,
    format_ms,
    format_params,
    format_shape,
)

# ---------------------------------------------------------------------------
# Additional fixtures for the training-loop / aggregation path.
# ---------------------------------------------------------------------------


class RepeatedForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))

# --- Fixtures ---


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.relu(x)
        return x


class DeadLayerModel(nn.Module):
    """Model with a layer that outputs all zeros."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)
        with torch.no_grad():
            self.linear2.weight.zero_()
            self.linear2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class NaNLayer(nn.Module):
    """Module that outputs all NaN values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, float("nan"))


class NaNModel(nn.Module):
    """Model that produces NaN outputs via a child module."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.nan_layer = NaNLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.nan_layer(x)
        return x


# --- Utils Tests ---


class TestComputeTensorStats:
    def test_normal_tensor(self) -> None:
        t = torch.randn(100, 100)
        stats = compute_tensor_stats(t)
        assert stats["mean"] is not None
        assert stats["std"] is not None
        assert abs(stats["mean"]) < 0.5
        assert abs(stats["std"] - 1.0) < 0.2
        assert stats["nan_count"] == 0
        assert stats["inf_count"] == 0

    def test_tensor_with_nans(self) -> None:
        t = torch.tensor([1.0, float("nan"), 3.0])
        stats = compute_tensor_stats(t)
        assert stats["nan_count"] == 1
        assert stats["mean"] is not None

    def test_tensor_with_infs(self) -> None:
        t = torch.tensor([1.0, float("inf"), 3.0])
        stats = compute_tensor_stats(t)
        assert stats["inf_count"] == 1

    def test_integer_tensor(self) -> None:
        t = torch.tensor([1, 2, 3, 4, 5])
        stats = compute_tensor_stats(t)
        assert stats["mean"] is None

    def test_empty_tensor(self) -> None:
        t = torch.tensor([])
        stats = compute_tensor_stats(t)
        assert stats["mean"] is None


class TestFormatters:
    def test_format_params(self) -> None:
        assert format_params(500) == "500"
        assert format_params(1500) == "1.5K"
        assert format_params(1_500_000) == "1.5M"
        assert format_params(2_000_000_000) == "2.0B"

    def test_format_shape(self) -> None:
        assert format_shape([8, 512, 768]) == "[8, 512, 768]"
        assert format_shape([10]) == "[10]"

    def test_format_ms(self) -> None:
        assert "ms" in format_ms(0.001)
        assert "ms" in format_ms(0.1)
        assert "ms" in format_ms(5.0)


# --- Collector Tests ---


class TestTensorStatsDataclass:
    def test_to_dict(self) -> None:
        stats = TensorStats(mean=0.5, std=1.0, min=-1.0, max=2.0, nan_count=0, inf_count=0)
        d = stats.to_dict()
        assert d["mean"] == 0.5
        assert d["std"] == 1.0


class TestTensorInfoDevice:
    def test_device_included_in_dict(self) -> None:
        info = TensorInfo(shape=[4, 10], dtype="float32", device="cuda:0")
        d = info.to_dict()
        assert d["device"] == "cuda:0"

    def test_device_defaults_to_empty(self) -> None:
        info = TensorInfo(shape=[4, 10], dtype="float32")
        d = info.to_dict()
        assert d["device"] == ""


class TestLayerTrace:
    def test_to_dict(self) -> None:
        trace = LayerTrace(
            name="linear",
            module_type="Linear",
            full_name="linear",
            depth=0,
            params=200,
        )
        d = trace.to_dict()
        assert d["name"] == "linear"
        assert d["type"] == "Linear"
        assert d["params"] == 200

    def test_gradient_fields_in_to_dict(self) -> None:
        trace = LayerTrace(
            name="linear",
            module_type="Linear",
            full_name="linear",
            depth=0,
            params=200,
            grad_norm=1.5,
            grad_mean=0.01,
            grad_has_nan=False,
            has_zero_grad=False,
        )
        d = trace.to_dict()
        assert d["grad_norm"] == 1.5
        assert d["grad_mean"] == 0.01
        assert d["grad_has_nan"] is False
        assert d["has_zero_grad"] is False


# --- Hook Tests ---


class TestHookManager:
    def test_register_unregister(self) -> None:
        model = SimpleModel()
        record = TraceRecord()
        manager = HookManager(model, record)
        manager.register()
        # 3 hooks per module: pre_hook, forward_hook, backward_hook
        assert len(manager._hooks) > 0
        manager.unregister()
        assert len(manager._hooks) == 0

    def test_capture_layer_data(self) -> None:
        model = SimpleModel()
        record = TraceRecord()
        manager = HookManager(model, record)
        manager.register()
        manager.enable()

        x = torch.randn(4, 10)
        _ = model(x)

        manager.disable()
        manager.unregister()

        assert len(record.layers) > 0
        names = [layer.name for layer in record.layers]
        assert "linear" in names
        assert "relu" in names

    def test_device_captured(self) -> None:
        model = SimpleModel()
        record = TraceRecord()
        manager = HookManager(model, record)
        manager.register()
        manager.enable()

        x = torch.randn(4, 10)
        _ = model(x)

        manager.disable()
        manager.unregister()

        for layer in record.layers:
            for info in layer.inputs + layer.outputs:
                assert info.device != ""

    def test_accurate_latency(self) -> None:
        """Pre-hook + post-hook should produce non-negative latency."""
        model = SimpleModel()
        record = TraceRecord()
        manager = HookManager(model, record)
        manager.register()
        manager.enable()

        x = torch.randn(4, 10)
        _ = model(x)

        manager.disable()
        manager.unregister()

        for layer in record.layers:
            assert layer.latency_ms >= 0

    def test_backward_hook_captures_gradients(self) -> None:
        model = SimpleModel()
        record = TraceRecord()
        manager = HookManager(model, record)
        manager.register()
        manager.enable()

        x = torch.randn(4, 10, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        manager.disable()
        manager.unregister()

        linear_layers = [lay for lay in record.layers if lay.module_type == "Linear"]
        assert len(linear_layers) > 0
        for layer in linear_layers:
            assert layer.grad_norm is not None
            assert layer.grad_norm >= 0


# --- Storage Tests ---


class TestStorage:
    def test_export_and_load_roundtrip(self) -> None:
        record = TraceRecord(
            metadata={"model_name": "test", "pytorch_version": "2.0"},
            layers=[
                LayerTrace(
                    name="linear",
                    module_type="Linear",
                    full_name="model.linear",
                    depth=1,
                    params=200,
                    inputs=[
                        TensorInfo(
                            shape=[4, 10],
                            dtype="float32",
                            device="cpu",
                            stats=TensorStats(mean=0.0, std=1.0),
                        )
                    ],
                    outputs=[
                        TensorInfo(
                            shape=[4, 20],
                            dtype="float32",
                            device="cpu",
                            stats=TensorStats(mean=0.1, std=0.5),
                        )
                    ],
                    latency_ms=0.5,
                    grad_norm=0.5,
                    grad_mean=0.01,
                )
            ],
            warnings=[{"type": "test", "severity": "info", "layer": "x", "message": "test"}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            export_json(record, path)

            loaded = load_json(path)
            assert len(loaded.layers) == 1
            assert loaded.layers[0].name == "linear"
            assert loaded.layers[0].inputs[0].stats is not None
            assert loaded.layers[0].inputs[0].stats.mean == 0.0
            assert loaded.layers[0].inputs[0].device == "cpu"
            assert loaded.layers[0].grad_norm == 0.5
            assert loaded.layers[0].grad_mean == 0.01
            assert loaded.metadata["model_name"] == "test"
            assert len(loaded.warnings) == 1


# --- Analyzer Tests ---


class TestAnalyzer:
    def test_detect_nan(self) -> None:
        record = TraceRecord(
            layers=[
                LayerTrace(
                    name="bad",
                    module_type="Linear",
                    full_name="bad",
                    depth=0,
                    params=0,
                    has_nan=True,
                )
            ]
        )
        anomalies = detect_anomalies(record)
        assert any(a.type == AnomalyType.NAN_ACTIVATION for a in anomalies)

    def test_detect_inf(self) -> None:
        record = TraceRecord(
            layers=[
                LayerTrace(
                    name="bad",
                    module_type="Linear",
                    full_name="bad",
                    depth=0,
                    params=0,
                    has_inf=True,
                )
            ]
        )
        anomalies = detect_anomalies(record)
        assert any(a.type == AnomalyType.INF_ACTIVATION for a in anomalies)

    def test_detect_dead_layer(self) -> None:
        record = TraceRecord(
            layers=[
                LayerTrace(
                    name="dead",
                    module_type="Linear",
                    full_name="dead",
                    depth=0,
                    params=0,
                    outputs=[
                        TensorInfo(
                            shape=[4, 10],
                            dtype="float32",
                            stats=TensorStats(mean=0.0, std=0.0),
                        )
                    ],
                )
            ]
        )
        anomalies = detect_anomalies(record)
        assert any(a.type == AnomalyType.DEAD_LAYER for a in anomalies)

    def test_detect_exploding_variance(self) -> None:
        record = TraceRecord(
            layers=[
                LayerTrace(
                    name="exploding",
                    module_type="Linear",
                    full_name="exploding",
                    depth=0,
                    params=0,
                    outputs=[
                        TensorInfo(
                            shape=[4, 10],
                            dtype="float32",
                            stats=TensorStats(mean=0.0, std=200.0),
                        )
                    ],
                )
            ]
        )
        anomalies = detect_anomalies(record)
        assert any(a.type == AnomalyType.EXPLODING_VARIANCE for a in anomalies)

    def test_detect_zero_gradient(self) -> None:
        record = TraceRecord(
            layers=[
                LayerTrace(
                    name="dead_grad",
                    module_type="Linear",
                    full_name="dead_grad",
                    depth=0,
                    params=100,
                    has_zero_grad=True,
                    grad_norm=0.0,
                )
            ]
        )
        anomalies = detect_anomalies(record)
        assert any(a.type == AnomalyType.ZERO_GRADIENT for a in anomalies)

    def test_detect_nan_gradient(self) -> None:
        record = TraceRecord(
            layers=[
                LayerTrace(
                    name="nan_grad",
                    module_type="Linear",
                    full_name="nan_grad",
                    depth=0,
                    params=100,
                    grad_has_nan=True,
                )
            ]
        )
        anomalies = detect_anomalies(record)
        assert any(a.type == AnomalyType.NAN_GRADIENT for a in anomalies)

    def test_no_anomalies_on_clean_data(self) -> None:
        record = TraceRecord(
            layers=[
                LayerTrace(
                    name="clean",
                    module_type="Linear",
                    full_name="clean",
                    depth=0,
                    params=100,
                    inputs=[
                        TensorInfo(
                            shape=[4, 10],
                            dtype="float32",
                            stats=TensorStats(mean=0.0, std=1.0),
                        )
                    ],
                    outputs=[
                        TensorInfo(
                            shape=[4, 10],
                            dtype="float32",
                            stats=TensorStats(mean=0.0, std=1.0),
                        )
                    ],
                )
            ]
        )
        anomalies = detect_anomalies(record)
        assert len(anomalies) == 0


class TestVarianceSpikeOrderIndependence:
    """The per-layer input->output spike check must not depend on execution
    ordering across siblings in a branched model."""

    def _branch_layer(
        self,
        full_name: str,
        in_std: float,
        out_std: float,
    ) -> LayerTrace:
        return LayerTrace(
            name=full_name.rsplit(".", 1)[-1],
            module_type="Linear",
            full_name=full_name,
            depth=0,
            params=200,
            inputs=[
                TensorInfo(
                    shape=[4, 10],
                    dtype="torch.float32",
                    device="cpu",
                    stats=TensorStats(mean=0.0, std=in_std),
                )
            ],
            outputs=[
                TensorInfo(
                    shape=[4, 10],
                    dtype="torch.float32",
                    device="cpu",
                    stats=TensorStats(mean=0.0, std=out_std),
                )
            ],
        )

    def test_unrelated_sibling_does_not_trigger_spike(self) -> None:
        # Two sibling layers; a.out has std 1.0, b.in has std 1.0, but b does
        # NOT feed off a. Old cross-layer code would compare a.out -> b.out
        # and (here) not flag; the more interesting case is below.
        record = TraceRecord(
            layers=[
                self._branch_layer("branch.a", in_std=1.0, out_std=1.0),
                self._branch_layer("branch.b", in_std=1.0, out_std=1.0),
            ]
        )
        anomalies = detect_anomalies(record)
        assert not any(a.type == AnomalyType.EXPLODING_VARIANCE for a in anomalies)

    def test_per_layer_input_to_output_spike_flagged(self) -> None:
        # A layer that amplifies std 10x between its own input and output.
        record = TraceRecord(
            layers=[self._branch_layer("amp", in_std=0.1, out_std=2.0)]
        )
        anomalies = detect_anomalies(record)
        spikes = [a for a in anomalies if a.type == AnomalyType.EXPLODING_VARIANCE]
        assert spikes, "expected per-layer variance spike to be flagged"
        details = spikes[0].details or {}
        assert details.get("input_std") == 0.1
        assert details.get("output_std") == 2.0
        assert details.get("ratio") == 20.0

    def test_meaningless_adjacent_layer_not_compared(self) -> None:
        # branch.a outputs std 1.0, branch.b takes an UNRELATED input with
        # std 1.0 and outputs std 1.0. Old cross-layer code would compare
        # a.out -> b.out (1.0 -> 1.0) and not flag -- but if a.out had been
        # much smaller (0.01) the old code would have falsely flagged b as
        # exploding relative to a, even though b never saw a's output.
        record = TraceRecord(
            layers=[
                self._branch_layer("branch.a", in_std=1.0, out_std=0.01),
                self._branch_layer("branch.b", in_std=1.0, out_std=1.0),
            ]
        )
        anomalies = detect_anomalies(record)
        # No spike for branch.b: its own 1.0 -> 1.0 is fine.
        spike_layers = {
            a.layer for a in anomalies if a.type == AnomalyType.EXPLODING_VARIANCE
        }
        assert "branch.b" not in spike_layers
        # branch.a's output (0.01) is smaller than its input (1.0): no spike.
        assert "branch.a" not in spike_layers


# --- Integration Tests ---


class TestTraceSession:
    def test_basic_trace(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            x = torch.randn(4, 10)
            output = model(x)

        assert output.shape == (4, 20)
        assert len(session.record.layers) > 0
        assert "pytorch_version" in session.record.metadata

    def test_metadata_includes_device_info(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            x = torch.randn(4, 10)
            _ = model(x)

        meta = session.record.metadata
        assert "cuda_available" in meta
        assert "active_device" in meta

    def test_export(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            x = torch.randn(4, 10)
            _ = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            result = session.export(path)
            assert result.exists()
            data = json.loads(result.read_text())
            assert "metadata" in data
            assert "layers" in data
            # Verify device is in exported tensor info
            first_layer = data["layers"][0]
            assert "device" in first_layer["outputs"][0]

    def test_summary(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            x = torch.randn(4, 10)
            _ = model(x)

        summary = session.summary()
        assert "SimpleModel" in summary
        assert "Layers traced" in summary

    def test_dead_layer_detection(self) -> None:
        model = DeadLayerModel()
        session = TraceSession(model)

        with session:
            x = torch.randn(4, 10)
            _ = model(x)

        assert len(session.anomalies) > 0
        assert any(a.type == AnomalyType.DEAD_LAYER for a in session.anomalies)

    def test_nan_detection(self) -> None:
        model = NaNModel()
        session = TraceSession(model)

        with session:
            x = torch.ones(4, 10)
            _ = model(x)

        assert any(a.type == AnomalyType.NAN_ACTIVATION for a in session.anomalies)

    def test_model_name_override(self) -> None:
        model = SimpleModel()
        session = TraceSession(model, model_name="MyCustomModel")

        with session:
            x = torch.randn(4, 10)
            _ = model(x)

        assert session.record.metadata["model_name"] == "MyCustomModel"

    def test_nested_model(self) -> None:
        class Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = SimpleModel()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x)

        model = Outer()
        session = TraceSession(model)

        with session:
            x = torch.randn(4, 10)
            _ = model(x)

        full_names = [layer.full_name for layer in session.record.layers]
        assert any("inner" in name for name in full_names)

    def test_backward_tracing(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            x = torch.randn(4, 10, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()

        linear_layers = [lay for lay in session.record.layers if lay.module_type == "Linear"]
        assert len(linear_layers) > 0
        for layer in linear_layers:
            assert layer.grad_norm is not None

    def test_zero_gradient_detection(self) -> None:
        """Model that outputs constant zero should produce zero gradients."""

        class ZeroOutputModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.linear1(x)
                x = self.linear2(x) * 0.0  # guaranteed zero output
                return x

        model = ZeroOutputModel()
        session = TraceSession(model)

        with session:
            x = torch.randn(4, 10, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()

        assert any(a.type == AnomalyType.ZERO_GRADIENT for a in session.anomalies)


# ---------------------------------------------------------------------------
# Aggregation / training-loop behaviour.
# ---------------------------------------------------------------------------


class TestRepeatedForwardsAggregation:
    def test_no_duplicate_layer_entries(self) -> None:
        model = RepeatedForwardModel()
        session = TraceSession(model)

        with session:
            for _ in range(5):
                model(torch.randn(4, 10))

        # Two child modules (linear + relu) => exactly two layer entries,
        # not ten.
        assert len(session.record.layers) == 2

    def test_forward_count_accumulates(self) -> None:
        model = RepeatedForwardModel()
        session = TraceSession(model)

        with session:
            for _ in range(7):
                model(torch.randn(4, 10))

        for layer in session.record.layers:
            assert layer.forward_count == 7

    def test_latency_aggregation_fields_populated(self) -> None:
        model = RepeatedForwardModel()
        session = TraceSession(model)

        with session:
            for _ in range(3):
                model(torch.randn(4, 10))

        linear = next(layer for layer in session.record.layers if layer.module_type == "Linear")
        assert linear.latency_ms_min is not None
        assert linear.latency_ms_max is not None
        assert linear.latency_ms_mean is not None
        assert linear.latency_ms_min <= linear.latency_ms_mean <= linear.latency_ms_max
        assert linear.latency_ms > 0

    def test_summary_mentions_forward_count(self) -> None:
        model = RepeatedForwardModel()
        session = TraceSession(model)

        with session:
            for _ in range(4):
                model(torch.randn(4, 10))

        summary = session.summary()
        assert "Forward passes: 4" in summary


# ---------------------------------------------------------------------------
# capture_stats gating.
# ---------------------------------------------------------------------------


class TestCaptureStatsFlag:
    def test_stats_omitted_when_disabled(self) -> None:
        model = SimpleModel()
        session = TraceSession(model, capture_stats=False)

        with session:
            model(torch.randn(4, 10))

        for layer in session.record.layers:
            for info in layer.inputs + layer.outputs:
                assert info.stats is None

    def test_stats_present_when_enabled(self) -> None:
        model = SimpleModel()
        session = TraceSession(model, capture_stats=True)

        with session:
            model(torch.randn(4, 10))

        for layer in session.record.layers:
            for info in layer.outputs:
                if info.dtype.startswith("torch.float"):
                    assert info.stats is not None
                    assert info.stats.mean is not None


# ---------------------------------------------------------------------------
# Re-entrancy / reset semantics.
# ---------------------------------------------------------------------------


class TestSessionReset:
    def test_reenter_clears_prior_layers(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            model(torch.randn(4, 10))
        n_first = len(session.record.layers)

        with session:
            model(torch.randn(4, 10))
        # Layer count should not double across two `with` blocks.
        assert len(session.record.layers) == n_first

    def test_reenter_resets_forward_counts(self) -> None:
        model = RepeatedForwardModel()
        session = TraceSession(model)

        with session:
            for _ in range(3):
                model(torch.randn(4, 10))
        counts_first = {lay.full_name: lay.forward_count for lay in session.record.layers}
        assert all(c == 3 for c in counts_first.values())

        # Re-enter: counts must reset, not continue accumulating from 3.
        with session:
            model(torch.randn(4, 10))
        for layer in session.record.layers:
            assert layer.forward_count == 1
            assert layer.latency_ms_min == layer.latency_ms_max == layer.latency_ms

    def test_reenter_clears_anomalies(self) -> None:
        model = NaNModel()
        session = TraceSession(model)

        with session:
            model(torch.ones(4, 10))
        assert any(a.type == AnomalyType.NAN_ACTIVATION for a in session.anomalies)

        # Re-entering clears the anomalies list; a clean session afterwards
        # produces no anomalies.
        clean_session = TraceSession(SimpleModel())
        with clean_session:
            clean_session._model(torch.randn(4, 10))
        assert len(clean_session.anomalies) == 0


# ---------------------------------------------------------------------------
# Storage round-trip with the new aggregation fields.
# ---------------------------------------------------------------------------


class TestStorageAggregationRoundtrip:
    def test_lat_aggregation_fields_survive_roundtrip(self) -> None:
        record = TraceRecord(
            metadata={"model_name": "agg"},
            layers=[
                LayerTrace(
                    name="linear",
                    module_type="Linear",
                    full_name="linear",
                    depth=0,
                    params=200,
                    latency_ms=0.4,
                    latency_ms_min=0.1,
                    latency_ms_max=0.9,
                    latency_ms_mean=0.5,
                    forward_count=3,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            export_json(record, path)
            loaded = load_json(path)

        layer = loaded.layers[0]
        assert layer.forward_count == 3
        assert layer.latency_ms_min == 0.1
        assert layer.latency_ms_max == 0.9
        assert layer.latency_ms_mean == 0.5
        assert layer.latency_ms == 0.4


# ---------------------------------------------------------------------------
# Diff / CI regression mode.
# ---------------------------------------------------------------------------


def _layer(
    full_name: str = "linear",
    *,
    module_type: str = "Linear",
    latency_ms_mean: float | None = 1.0,
    latency_ms: float = 1.0,
    forward_count: int = 1,
    grad_norm: float | None = None,
    out_std: float | None = None,
    out_shape: tuple[int, ...] = (4, 10),
) -> LayerTrace:
    outputs: list[TensorInfo] = []
    if out_std is not None or out_shape is not None:
        outputs.append(
            TensorInfo(
                shape=list(out_shape),
                dtype="torch.float32",
                device="cpu",
                stats=TensorStats(mean=0.0, std=out_std),
            )
        )
    return LayerTrace(
        name=full_name.rsplit(".", 1)[-1],
        module_type=module_type,
        full_name=full_name,
        depth=0,
        params=200,
        latency_ms=latency_ms,
        latency_ms_mean=latency_ms_mean,
        forward_count=forward_count,
        grad_norm=grad_norm,
        outputs=outputs,
    )


class TestDiffRecords:
    def test_identical_traces_have_no_regressions(self) -> None:
        baseline = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", latency_ms_mean=1.0)],
        )
        current = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", latency_ms_mean=1.0)],
        )
        diff = diff_records(baseline, current)
        assert not diff.has_regressions
        assert diff.entries == []

    def test_latency_regression_flagged(self) -> None:
        baseline = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", latency_ms_mean=1.0)],
        )
        current = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", latency_ms_mean=1.5)],  # +50%
        )
        diff = diff_records(baseline, current, latency_threshold=0.1)
        assert diff.has_regressions
        kinds = [e.kind for e in diff.entries]
        assert "latency_regression" in kinds

    def test_latency_improvement_is_info(self) -> None:
        baseline = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", latency_ms_mean=2.0)],
        )
        current = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", latency_ms_mean=1.0)],  # -50%
        )
        diff = diff_records(baseline, current, latency_threshold=0.1)
        assert not diff.has_regressions
        entry = next(e for e in diff.entries if e.kind == "latency_improvement")
        assert entry.severity == DiffSeverity.INFO

    def test_layer_added_removed(self) -> None:
        baseline = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a"), _layer("b")],
        )
        current = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a"), _layer("c")],
        )
        diff = diff_records(baseline, current)
        kinds = {e.kind for e in diff.entries}
        assert "layer_removed" in kinds  # b removed
        assert "layer_added" in kinds  # c added
        assert diff.has_regressions

    def test_shape_change_is_critical(self) -> None:
        baseline = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", out_shape=(4, 10))],
        )
        current = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", out_shape=(4, 20))],
        )
        diff = diff_records(baseline, current)
        entry = next(e for e in diff.entries if e.kind == "shape_change")
        assert entry.severity == DiffSeverity.CRITICAL
        assert diff.critical_count >= 1

    def test_new_anomaly_is_critical(self) -> None:
        baseline = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a")],
            warnings=[],
        )
        current = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", out_std=200.0)],
            warnings=[
                {
                    "type": "exploding_variance",
                    "severity": "warning",
                    "layer": "a",
                    "message": "std exploded",
                }
            ],
        )
        diff = diff_records(baseline, current)
        entry = next(e for e in diff.entries if e.kind == "new_anomaly")
        # severity mirrors the warning's severity; here warning
        assert entry.severity == DiffSeverity.WARNING
        assert diff.has_regressions

    def test_resolved_anomaly_is_info(self) -> None:
        baseline = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", out_std=200.0)],
            warnings=[
                {
                    "type": "exploding_variance",
                    "severity": "warning",
                    "layer": "a",
                    "message": "std exploded",
                }
            ],
        )
        current = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", out_std=1.0)],
            warnings=[],
        )
        diff = diff_records(baseline, current)
        entry = next(e for e in diff.entries if e.kind == "resolved_anomaly")
        assert entry.severity == DiffSeverity.INFO

    def test_to_dict_serializable(self) -> None:
        baseline = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", latency_ms_mean=1.0)],
        )
        current = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", latency_ms_mean=2.0)],
        )
        diff = diff_records(baseline, current)
        d = diff.to_dict()
        assert d["baseline_model"] == "m"
        assert d["has_regressions"] is True
        assert isinstance(d["entries"], list)
        assert any(e["kind"] == "latency_regression" for e in d["entries"])

    def test_diff_via_cli_json_exits_nonzero_on_regression(self, tmp_path: Path) -> None:
        baseline = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", latency_ms_mean=1.0)],
        )
        current = TraceRecord(
            metadata={"model_name": "m"},
            layers=[_layer("a", latency_ms_mean=3.0)],
        )
        base_path = tmp_path / "base.json"
        curr_path = tmp_path / "curr.json"
        export_json(baseline, base_path)
        export_json(current, curr_path)

        # Programmatic diff matches a fresh load+diff round-trip.
        loaded_base = load_json(base_path)
        loaded_curr = load_json(curr_path)
        diff = diff_records(loaded_base, loaded_curr)
        assert diff.has_regressions


# ---------------------------------------------------------------------------
# GPU memory tracking (#8).
# ---------------------------------------------------------------------------


class TestFormatBytes:
    def test_none(self) -> None:
        assert format_bytes(None) == "-"

    def test_zero(self) -> None:
        assert format_bytes(0.0) == "0B"

    def test_positive_scaling(self) -> None:
        assert "KiB" in format_bytes(2048.0)
        assert "MiB" in format_bytes(5 * 1024**2)
        assert "GiB" in format_bytes(3 * 1024**3)

    def test_negative_shown(self) -> None:
        # Memory release deltas surface as negative.
        out = format_bytes(-2048.0)
        assert out.startswith("-")
        assert "KiB" in out


class TestMemoryFieldsOnCPU:
    """On CPU traces (the default in CI), memory fields stay None."""

    def test_cpu_trace_leaves_mem_fields_none(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            model(torch.randn(4, 10))

        for layer in session.record.layers:
            assert layer.fwd_mem_alloc_delta is None
            assert layer.fwd_mem_reserved_delta is None
            assert layer.bwd_mem_alloc_delta is None
            assert layer.bwd_mem_reserved_delta is None

    def test_summary_omits_mem_section_on_cpu(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            model(torch.randn(4, 10))

        summary = session.summary()
        assert "Heaviest fwd mem" not in summary
        assert "Heaviest bwd mem" not in summary


class TestMemoryFieldsRoundtrip:
    def test_mem_deltas_survive_roundtrip(self) -> None:
        record = TraceRecord(
            metadata={"model_name": "m"},
            layers=[
                LayerTrace(
                    name="a",
                    module_type="Linear",
                    full_name="a",
                    depth=0,
                    params=200,
                    latency_ms=1.0,
                    fwd_mem_alloc_delta=1.5 * 1024**2,
                    fwd_mem_reserved_delta=2.0 * 1024**2,
                    bwd_mem_alloc_delta=0.5 * 1024**2,
                    bwd_mem_reserved_delta=1.0 * 1024**2,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            export_json(record, path)
            loaded = load_json(path)

        layer = loaded.layers[0]
        assert layer.fwd_mem_alloc_delta == 1.5 * 1024**2
        assert layer.fwd_mem_reserved_delta == 2.0 * 1024**2
        assert layer.bwd_mem_alloc_delta == 0.5 * 1024**2
        assert layer.bwd_mem_reserved_delta == 1.0 * 1024**2


class TestMemoryFieldsSummary:
    def test_summary_lists_heaviest_fwd_mem_when_present(self) -> None:
        record = TraceRecord(
            metadata={"model_name": "m", "total_time_ms": 1.0},
            layers=[
                LayerTrace(
                    name="a",
                    module_type="Linear",
                    full_name="a",
                    depth=0,
                    params=200,
                    latency_ms=0.5,
                    latency_ms_mean=0.5,
                    forward_count=1,
                    fwd_mem_alloc_delta=5 * 1024**2,
                    fwd_mem_reserved_delta=8 * 1024**2,
                ),
                LayerTrace(
                    name="b",
                    module_type="Linear",
                    full_name="b",
                    depth=0,
                    params=200,
                    latency_ms=0.3,
                    latency_ms_mean=0.3,
                    forward_count=1,
                    fwd_mem_alloc_delta=1 * 1024**2,
                    fwd_mem_reserved_delta=2 * 1024**2,
                ),
            ],
        )
        # Build a session-like view: we can't easily construct a TraceSession
        # without running hooks, so drive summary() via a thin wrapper.
        session = TraceSession.__new__(TraceSession)
        session._model_name = "m"
        session._record = record
        session._anomalies = []
        summary = session.summary()
        assert "Heaviest fwd mem" in summary
        assert "a" in summary.split("Heaviest fwd mem", 1)[1].splitlines()[0]
        assert "MiB" in summary


# ---------------------------------------------------------------------------
# Configurable thresholds (#11).
# ---------------------------------------------------------------------------


class TestThresholds:
    def _dead_layer_record(self, mean: float, std: float) -> TraceRecord:
        return TraceRecord(
            layers=[
                LayerTrace(
                    name="l",
                    module_type="Linear",
                    full_name="l",
                    depth=0,
                    params=0,
                    outputs=[
                        TensorInfo(
                            shape=[4, 10],
                            dtype="float32",
                            stats=TensorStats(mean=mean, std=std),
                        )
                    ],
                )
            ]
        )

    def test_default_dead_layer_threshold_does_not_flag_tiny_but_above_1e6(self) -> None:
        # std just above the new default of 1e-6 should not flag.
        record = self._dead_layer_record(mean=0.0, std=1e-5)
        anomalies = detect_anomalies(record)
        assert not any(a.type == AnomalyType.DEAD_LAYER for a in anomalies)

    def test_custom_dead_layer_threshold_can_tighten(self) -> None:
        # With a tighter threshold (1e-12), a layer at std=1e-5 is not flagged.
        record = self._dead_layer_record(mean=0.0, std=1e-5)
        anomalies = detect_anomalies(
            record, thresholds=Thresholds(dead_layer_mean=1e-12, dead_layer_std=1e-12)
        )
        assert not any(a.type == AnomalyType.DEAD_LAYER for a in anomalies)

    def test_custom_dead_layer_threshold_can_loosen(self) -> None:
        # Loosening the threshold flags a previously-clean layer.
        record = self._dead_layer_record(mean=0.0, std=0.05)
        default_anomalies = detect_anomalies(record)
        assert not any(a.type == AnomalyType.DEAD_LAYER for a in default_anomalies)

        anomalies = detect_anomalies(
            record, thresholds=Thresholds(dead_layer_mean=1.0, dead_layer_std=0.1)
        )
        assert any(a.type == AnomalyType.DEAD_LAYER for a in anomalies)

    def test_exploding_variance_threshold_is_configurable(self) -> None:
        record = TraceRecord(
            layers=[
                LayerTrace(
                    name="l",
                    module_type="Linear",
                    full_name="l",
                    depth=0,
                    params=0,
                    outputs=[
                        TensorInfo(
                            shape=[4, 10],
                            dtype="float32",
                            stats=TensorStats(mean=0.0, std=20.0),
                        )
                    ],
                )
            ]
        )
        # Default threshold is 100; std=20 is not exploding by default.
        assert not any(
            a.type == AnomalyType.EXPLODING_VARIANCE for a in detect_anomalies(record)
        )
        # Lower the threshold to 10 and it now flags.
        anomalies = detect_anomalies(
            record,
            thresholds=Thresholds(exploding_variance_std=10.0, high_variance_min=5.0),
        )
        assert any(a.type == AnomalyType.EXPLODING_VARIANCE for a in anomalies)

    def test_variance_spike_ratio_is_configurable(self) -> None:
        record = TraceRecord(
            layers=[
                LayerTrace(
                    name="l",
                    module_type="Linear",
                    full_name="l",
                    depth=0,
                    params=0,
                    inputs=[
                        TensorInfo(
                            shape=[4, 10],
                            dtype="float32",
                            stats=TensorStats(mean=0.0, std=0.1),
                        )
                    ],
                    outputs=[
                        TensorInfo(
                            shape=[4, 10],
                            dtype="float32",
                            stats=TensorStats(mean=0.0, std=0.5),
                        )
                    ],
                )
            ]
        )
        # Default ratio is 10; 0.1 -> 0.5 is a 5x ratio, not flagged.
        assert not any(
            a.type == AnomalyType.EXPLODING_VARIANCE for a in detect_anomalies(record)
        )
        # Lower the ratio to 3x and it flags.
        anomalies = detect_anomalies(
            record, thresholds=Thresholds(variance_spike_ratio=3.0)
        )
        assert any(a.type == AnomalyType.EXPLODING_VARIANCE for a in anomalies)


class TestSessionAcceptsThresholds:
    def test_thresholds_propagate_to_session_anomalies(self) -> None:
        # Construct a model that produces a layer with std around 0.5; with
        # the default dead-layer threshold it stays clean, but with a very
        # loose threshold it should be flagged through TraceSession.
        model = SimpleModel()
        loose = Thresholds(dead_layer_mean=1.0, dead_layer_std=1.0)
        session = TraceSession(model, thresholds=loose)

        with session:
            model(torch.randn(4, 10))

        assert any(a.type == AnomalyType.DEAD_LAYER for a in session.anomalies)


# ---------------------------------------------------------------------------
# Backward latency + grad_in capture (#9).
# ---------------------------------------------------------------------------


class TestBackwardLatency:
    def test_backward_latency_populated_after_backward(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            x = torch.randn(4, 10, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()

        linear = next(
            layer for layer in session.record.layers if layer.module_type == "Linear"
        )
        assert linear.backward_count >= 1
        assert linear.bwd_latency_ms > 0
        assert linear.bwd_latency_ms_min is not None
        assert linear.bwd_latency_ms_max is not None
        assert linear.bwd_latency_ms_mean is not None
        assert linear.bwd_latency_ms_min <= linear.bwd_latency_ms_mean
        assert linear.bwd_latency_ms_mean <= linear.bwd_latency_ms_max

    def test_backward_latency_zero_without_backward(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            model(torch.randn(4, 10))

        for layer in session.record.layers:
            assert layer.backward_count == 0
            assert layer.bwd_latency_ms == 0.0
            assert layer.bwd_latency_ms_mean is None

    def test_backward_latency_aggregates_across_steps(self) -> None:
        model = RepeatedForwardModel()
        session = TraceSession(model)

        with session:
            for _ in range(3):
                x = torch.randn(4, 10, requires_grad=True)
                out = model(x)
                out.sum().backward()

        for layer in session.record.layers:
            assert layer.backward_count == 3
            assert (layer.bwd_latency_ms_mean or 0) > 0


class TestGradInCapture:
    def test_grad_in_populated_after_backward(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            x = torch.randn(4, 10, requires_grad=True)
            out = model(x)
            out.sum().backward()

        linear = next(
            layer for layer in session.record.layers if layer.module_type == "Linear"
        )
        assert linear.grad_in_norm is not None
        assert linear.grad_in_norm >= 0
        assert linear.grad_in_mean is not None

    def test_grad_in_unset_without_backward(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            model(torch.randn(4, 10))

        for layer in session.record.layers:
            assert layer.grad_in_norm is None
            assert layer.grad_in_mean is None
            assert layer.grad_in_has_nan is False


class TestBackwardFieldsRoundtrip:
    def test_backward_fields_survive_roundtrip(self) -> None:
        record = TraceRecord(
            metadata={"model_name": "m"},
            layers=[
                LayerTrace(
                    name="a",
                    module_type="Linear",
                    full_name="a",
                    depth=0,
                    params=200,
                    latency_ms=0.5,
                    bwd_latency_ms=0.7,
                    bwd_latency_ms_min=0.4,
                    bwd_latency_ms_max=0.9,
                    bwd_latency_ms_mean=0.65,
                    backward_count=2,
                    grad_in_norm=0.8,
                    grad_in_mean=0.001,
                    grad_in_has_nan=False,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            export_json(record, path)
            loaded = load_json(path)

        layer = loaded.layers[0]
        assert layer.backward_count == 2
        assert layer.bwd_latency_ms == 0.7
        assert layer.bwd_latency_ms_min == 0.4
        assert layer.bwd_latency_ms_max == 0.9
        assert layer.bwd_latency_ms_mean == 0.65
        assert layer.grad_in_norm == 0.8
        assert layer.grad_in_mean == 0.001


class TestSummaryBackwardLatency:
    def test_summary_lists_slowest_backward_when_present(self) -> None:
        record = TraceRecord(
            metadata={"model_name": "m", "total_time_ms": 1.0},
            layers=[
                LayerTrace(
                    name="a",
                    module_type="Linear",
                    full_name="a",
                    depth=0,
                    params=200,
                    latency_ms=0.5,
                    latency_ms_mean=0.5,
                    forward_count=1,
                    bwd_latency_ms=2.0,
                    bwd_latency_ms_min=2.0,
                    bwd_latency_ms_max=2.0,
                    bwd_latency_ms_mean=2.0,
                    backward_count=1,
                ),
                LayerTrace(
                    name="b",
                    module_type="Linear",
                    full_name="b",
                    depth=0,
                    params=200,
                    latency_ms=0.3,
                    latency_ms_mean=0.3,
                    forward_count=1,
                    bwd_latency_ms=1.0,
                    bwd_latency_ms_min=1.0,
                    bwd_latency_ms_max=1.0,
                    bwd_latency_ms_mean=1.0,
                    backward_count=1,
                ),
            ],
        )
        session = TraceSession.__new__(TraceSession)
        session._model_name = "m"
        session._record = record
        session._anomalies = []
        summary = session.summary()
        assert "Slowest backward" in summary
        backward_line = next(
            line for line in summary.splitlines() if "Slowest backward" in line
        )
        assert "a" in backward_line


# ---------------------------------------------------------------------------
# JSON-safe metadata (#12) + re-entrancy guard (#13).
# ---------------------------------------------------------------------------


class TestJsonSafeMetadata:
    def test_non_native_metadata_value_is_stringified(self) -> None:
        record = TraceRecord(
            metadata={
                "model_name": "m",
                "custom_device": torch.device("cpu"),
                "custom_dtype": torch.float32,
            },
            layers=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            export_json(record, path)
            import json as _json

            data = _json.loads(path.read_text())
            assert data["metadata"]["model_name"] == "m"
            assert isinstance(data["metadata"]["custom_device"], str)
            assert "cpu" in data["metadata"]["custom_device"]


class TestDoubleEnterGuard:
    def test_double_enter_raises(self) -> None:
        model = SimpleModel()
        session = TraceSession(model)

        session.__enter__()
        try:
            with pytest.raises(RuntimeError, match="already active"):
                session.__enter__()
        finally:
            session.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# CLI filter / JSON / summary subcommand (#7).
# ---------------------------------------------------------------------------


class TestFilterOptions:
    def _layer(
        self,
        full_name: str = "a",
        module_type: str = "Linear",
        latency_ms_mean: float = 1.0,
    ) -> LayerTrace:
        return LayerTrace(
            name=full_name.rsplit(".", 1)[-1],
            module_type=module_type,
            full_name=full_name,
            depth=0,
            params=200,
            latency_ms_mean=latency_ms_mean,
            latency_ms=latency_ms_mean,
        )

    def test_no_filter_matches_everything(self) -> None:
        from tracetorch.cli import FilterOptions

        opts = FilterOptions()
        assert opts.matches(self._layer("a", "Linear", 1.0))

    def test_name_glob(self) -> None:
        from tracetorch.cli import FilterOptions

        opts = FilterOptions(name_pattern="encoder.*")
        assert opts.matches(self._layer("encoder.linear"))
        assert not opts.matches(self._layer("decoder.linear"))

    def test_module_type(self) -> None:
        from tracetorch.cli import FilterOptions

        opts = FilterOptions(module_type="Linear")
        assert opts.matches(self._layer("a", "Linear"))
        assert not opts.matches(self._layer("a", "ReLU"))

    def test_min_latency(self) -> None:
        from tracetorch.cli import FilterOptions

        opts = FilterOptions(min_latency_ms=0.5)
        assert opts.matches(self._layer("a", "Linear", latency_ms_mean=1.0))
        assert not opts.matches(self._layer("a", "Linear", latency_ms_mean=0.1))

    def test_combined_filters(self) -> None:
        from tracetorch.cli import FilterOptions

        opts = FilterOptions(
            name_pattern="enc*", module_type="Linear", min_latency_ms=0.5
        )
        assert opts.matches(self._layer("enc1", "Linear", latency_ms_mean=1.0))
        assert not opts.matches(self._layer("enc1", "ReLU", latency_ms_mean=1.0))
        assert not opts.matches(self._layer("enc1", "Linear", latency_ms_mean=0.1))
        assert not opts.matches(self._layer("dec1", "Linear", latency_ms_mean=1.0))


class TestCLICommands:
    def test_summary_command_prints_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tracetorch.cli import summary_command

        record = TraceRecord(
            metadata={"model_name": "demo", "total_time_ms": 5.0},
            layers=[
                LayerTrace(
                    name="a",
                    module_type="Linear",
                    full_name="a",
                    depth=0,
                    params=200,
                    latency_ms=1.0,
                    latency_ms_mean=1.0,
                    forward_count=1,
                )
            ],
        )
        path = tmp_path / "trace.json"
        export_json(record, path)

        summary_command(str(path))
        out = capsys.readouterr().out
        assert "demo" in out

    def test_inspect_command_json_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tracetorch.cli import FilterOptions, inspect_command

        record = TraceRecord(
            metadata={"model_name": "demo", "total_time_ms": 5.0},
            layers=[
                LayerTrace(
                    name="keep",
                    module_type="Linear",
                    full_name="keep",
                    depth=0,
                    params=200,
                    latency_ms=1.0,
                    latency_ms_mean=1.0,
                    forward_count=1,
                ),
                LayerTrace(
                    name="drop",
                    module_type="ReLU",
                    full_name="drop",
                    depth=0,
                    params=0,
                    latency_ms=0.05,
                    latency_ms_mean=0.05,
                    forward_count=1,
                ),
            ],
        )
        path = tmp_path / "trace.json"
        export_json(record, path)

        inspect_command(
            str(path),
            filter_opts=FilterOptions(min_latency_ms=0.5),
            json_output=True,
        )
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["metadata"]["model_name"] == "demo"
        names = [layer["full_name"] for layer in data["layers"]]
        assert names == ["keep"]
        assert "warnings" in data

    def test_inspect_command_filter_no_match(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tracetorch.cli import FilterOptions, inspect_command

        record = TraceRecord(
            metadata={"model_name": "demo"},
            layers=[
                LayerTrace(
                    name="a",
                    module_type="Linear",
                    full_name="a",
                    depth=0,
                    params=200,
                    latency_ms=0.05,
                    latency_ms_mean=0.05,
                    forward_count=1,
                )
            ],
        )
        path = tmp_path / "trace.json"
        export_json(record, path)

        inspect_command(
            str(path),
            filter_opts=FilterOptions(min_latency_ms=10.0),
        )
        out = capsys.readouterr().out
        assert "No layers matched the filter" in out


# ---------------------------------------------------------------------------
# Session watch mode (live flush during training).
# ---------------------------------------------------------------------------


class TestSessionWatchMode:
    def test_watch_file_written_during_session(self, tmp_path: Path) -> None:
        """The watch file should appear while the session is still active."""
        watch_path = tmp_path / "live.json"
        model = RepeatedForwardModel()
        session = TraceSession(
            model,
            watch_path=watch_path,
            watch_interval_s=0.1,
        )

        with session:
            # Run a few forwards to populate layers.
            for _ in range(3):
                model(torch.randn(4, 10))
            # Give the watch thread time to flush at least once.
            time.sleep(0.3)
            # The file should exist while still inside `with`.
            assert watch_path.exists(), "watch file was not written during session"

            # Verify the partial trace is loadable and has layer data.
            partial = load_json(watch_path)
            assert len(partial.layers) > 0
            # forward_count should reflect the 3 forwards we ran.
            for layer in partial.layers:
                assert layer.forward_count == 3

    def test_watch_file_final_flush_has_anomalies(self, tmp_path: Path) -> None:
        """After __exit__, the watch file should include warnings + total_time."""
        watch_path = tmp_path / "live.json"
        model = DeadLayerModel()
        session = TraceSession(
            model,
            watch_path=watch_path,
            watch_interval_s=0.1,
        )

        with session:
            model(torch.randn(4, 10))

        # Final flush on __exit__ should have written the complete trace.
        assert watch_path.exists()
        final = load_json(watch_path)
        assert final.metadata.get("total_time_ms", 0) > 0
        assert len(final.warnings) > 0  # dead layer anomaly

    def test_no_watch_without_watch_path(self, tmp_path: Path) -> None:
        """Without watch_path, no file should be written during the session."""
        model = SimpleModel()
        session = TraceSession(model)

        with session:
            model(torch.randn(4, 10))

        # No stray files.
        assert not (tmp_path / "live.json").exists()

    def test_watch_interval_clamped(self) -> None:
        """watch_interval_s below 0.1 is clamped to 0.1."""
        model = SimpleModel()
        session = TraceSession(model, watch_path="/tmp/dummy.json", watch_interval_s=0.001)
        assert session._watch_interval_s == 0.1


# ---------------------------------------------------------------------------
# CLI watch command.
# ---------------------------------------------------------------------------


class TestCLIWatchCommand:
    def test_watch_renders_initial_file(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """watch_command should render on the first cycle if the file exists."""
        from tracetorch.cli import watch_command

        record = TraceRecord(
            metadata={"model_name": "demo", "total_time_ms": 5.0},
            layers=[
                LayerTrace(
                    name="a",
                    module_type="Linear",
                    full_name="a",
                    depth=0,
                    params=200,
                    latency_ms=1.0,
                    latency_ms_mean=1.0,
                    forward_count=1,
                )
            ],
        )
        path = tmp_path / "trace.json"
        export_json(record, path)

        # Patch time.sleep to raise KeyboardInterrupt after one cycle so the
        # test doesn't block in the polling loop.
        call_count = 0

        def fake_sleep(_seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise KeyboardInterrupt()

        monkeypatch.setattr("tracetorch.cli.time.sleep", fake_sleep)

        watch_command(str(path), interval=0.01)
        out = capsys.readouterr().out
        assert "demo" in out
        assert "Stopped watching" in out

    def test_watch_picks_up_file_change(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """watch_command should re-render when the file mtime changes."""
        from tracetorch.cli import watch_command

        path = tmp_path / "trace.json"

        # Write version 1.
        record_v1 = TraceRecord(
            metadata={"model_name": "v1", "total_time_ms": 1.0},
            layers=[
                LayerTrace(
                    name="a",
                    module_type="Linear",
                    full_name="a",
                    depth=0,
                    params=200,
                    latency_ms=0.5,
                    latency_ms_mean=0.5,
                    forward_count=1,
                )
            ],
        )
        export_json(record_v1, path)

        cycle = 0

        def fake_sleep(_seconds: float) -> None:
            nonlocal cycle
            cycle += 1
            if cycle == 2:
                # Rewrite the file with version 2 before the next poll.
                record_v2 = TraceRecord(
                    metadata={"model_name": "v2_updated", "total_time_ms": 2.0},
                    layers=[
                        LayerTrace(
                            name="b",
                            module_type="ReLU",
                            full_name="b",
                            depth=0,
                            params=0,
                            latency_ms=0.1,
                            latency_ms_mean=0.1,
                            forward_count=1,
                        )
                    ],
                )
                export_json(record_v2, path)
            if cycle >= 4:
                raise KeyboardInterrupt()

        monkeypatch.setattr("tracetorch.cli.time.sleep", fake_sleep)

        watch_command(str(path), interval=0.01)
        out = capsys.readouterr().out
        assert "v1" in out
        assert "v2_updated" in out

    def test_watch_waits_for_missing_file(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """watch_command should wait if the file doesn't exist yet."""
        from tracetorch.cli import watch_command

        path = tmp_path / "not_yet.json"

        call_count = 0

        def fake_sleep(_seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise KeyboardInterrupt()

        monkeypatch.setattr("tracetorch.cli.time.sleep", fake_sleep)

        watch_command(str(path), interval=0.01)
        out = capsys.readouterr().out
        assert "Waiting for" in out
        assert "Stopped" in out
