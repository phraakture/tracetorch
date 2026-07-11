"""Tests for TraceTorch core functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from tracetorch import TraceSession
from tracetorch.analyzer import AnomalyType, detect_anomalies
from tracetorch.collector import LayerTrace, TensorInfo, TensorStats, TraceRecord
from tracetorch.hooks import HookManager
from tracetorch.storage import export_json, load_json
from tracetorch.utils import compute_tensor_stats, format_ms, format_params, format_shape

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
