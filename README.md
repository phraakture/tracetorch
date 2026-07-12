<p align="center">
  <img src="assets/header.jpg?v=3" alt="TraceTorch" width="100%">
</p>

A forensic trace console for PyTorch models.

## Features

- **Zero-code tracing** -- wraps any `nn.Module` via context manager, no hook scattering
- **Per-layer capture** -- name, type, shape, dtype, device, params, latency
- **Activation statistics** -- mean, std, min, max, NaN count, Inf count per tensor
- **Backward gradient tracing** -- grad norm, mean, NaN detection, zero-gradient detection
- **Anomaly detection** -- 8 built-in checks: NaN, Inf, dead layers, exploding variance, variance spikes, zero gradients, NaN gradients, empty outputs
- **JSON export** -- structured, flat format designed for dashboards and CI pipelines
- **Rich CLI inspector** -- nested layer tree, anomaly warnings, model summary in the terminal
- **Accurate latency** -- pre-hook to post-hook timing, not just stat-computation overhead

<p>
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/pypi/v/tracetorch-ml" alt="PyPI">
  <img src="https://img.shields.io/github/license/phraakture/tracetorch?v=1" alt="License">
  </p>

## Install

```bash
pip install tracetorch-ml
```

Requires Python 3.11+. Runtime dependencies: PyTorch 2.0+, Rich 13.0+.

## Quick start

```python
import torch
import torch.nn as nn
from tracetorch import TraceSession

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

session = TraceSession(model)

with session:
    x = torch.randn(32, 784)
    y = model(x)

print(session.summary())
session.export("trace.json")
```

Inspect the saved trace:

```bash
tracetorch inspect trace.json
```

<p align="center">
  <img src="assets/ss.jpg?v=2" alt="TraceTorch CLI output" width="100%">
</p>

## What gets captured

For each child module reached through `model.named_modules()`:

- module name, full name, type, and nesting depth
- direct parameter count (`recurse=False`)
- input and output tensor shape, dtype, and device
- floating tensor statistics: mean, std, min, max, NaN count, Inf count
- forward latency measured from pre-hook to post-hook
- backward gradient norm, mean, NaN flag, zero-gradient flag

Root model is not traced; only its children are.

## Backward tracing

Run backward inside the session to capture gradient data:

```python
session = TraceSession(model)

with session:
    x = torch.randn(4, 10, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

for layer in session.record.layers:
    print(layer.full_name, layer.grad_norm, layer.has_zero_grad)
```

Gradient fields stay `None` / `False` if no backward pass runs.

## Anomaly checks

| Type | Trigger |
| --- | --- |
| `nan_activation` | Output tensor contains NaN |
| `inf_activation` | Output tensor contains Inf |
| `dead_layer` | Output mean and std both near zero |
| `exploding_variance` | Std > 100, or >10x spike from previous layer |
| `high_variance` | Std between 10 and 100 |
| `empty_output` | No tensor output captured |
| `zero_gradient` | Backward gradient is all zeros |
| `nan_gradient` | Backward gradient contains NaN |

## JSON format

```json
{
  "metadata": {
    "model_name": "Sequential",
    "pytorch_version": "2.x",
    "python_version": "3.11.x",
    "cuda_available": false,
    "active_device": "cpu",
    "total_time_ms": 1.23
  },
  "layers": [
    {
      "name": "0",
      "type": "Linear",
      "full_name": "0",
      "depth": 1,
      "params": 200960,
      "inputs": [
        {
          "shape": [32, 784],
          "dtype": "torch.float32",
          "device": "cpu",
          "mean": 0.0,
          "std": 1.0,
          "min": -3.0,
          "max": 3.0,
          "nan_count": 0,
          "inf_count": 0
        }
      ],
      "outputs": [
        {
          "shape": [32, 256],
          "dtype": "torch.float32",
          "device": "cpu",
          "mean": 0.01,
          "std": 0.58,
          "min": -2.1,
          "max": 2.4,
          "nan_count": 0,
          "inf_count": 0
        }
      ],
      "latency_ms": 0.25,
      "grad_norm": null,
      "grad_mean": null,
      "grad_has_nan": false,
      "has_zero_grad": false
    }
  ],
  "warnings": []
}
```

Load a saved trace in Python:

```python
from tracetorch.storage import load_json

record = load_json("trace.json")
print(record.metadata)
print(record.layers[0].outputs)
```

## API Reference

### TraceSession

```python
session = TraceSession(model, capture_stats=True, model_name=None)
```

Context manager for one traced execution window.

| Property / Method | Description |
| --- | --- |
| `record` | Collected `TraceRecord` |
| `anomalies` | `list[Anomaly]` populated after exit |
| `export(path)` | Write JSON, return `Path` |
| `summary()` | Compact text summary |

### Data structures

```python
from tracetorch import TraceRecord, LayerTrace, TensorStats, Anomaly
```

All dataclasses serialize via `.to_dict()`. JSON format is flat for dashboard consumption.

### CLI

```bash
tracetorch inspect trace.json
```

Prints model summary, nested layer tree with tensor shapes/params/latency/std, and anomaly warnings.

## Architecture

```
PyTorch Model
      │
  PyTorch Hooks
  (forward_pre, forward, backward)
      │
  Trace Collector
  capture shapes, stats, grads
      │
  Anomaly Detector
  8 built-in checks
      │
  Storage Layer
  JSON export / import
      │
  CLI Inspector
  rich terminal output
```

## Development

```bash
git clone https://github.com/phraakture/tracetorch.git
cd tracetorch
pip install -e ".[dev]"

pytest                        # tests
ruff check tracetorch tests   # lint
mypy tracetorch               # types
```

Run the example:

```bash
python example.py
tracetorch inspect trace.json
```

## License

MIT
