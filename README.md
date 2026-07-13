<p align="center">
  <img src="assets/header.jpg?v=3" alt="TraceTorch" width="100%">
</p>

A forensic trace console for PyTorch models.

## Features

- **Zero-code tracing** -- wraps any `nn.Module` via context manager, no hook scattering
- **Per-layer capture** -- name, type, shape, dtype, device, params, latency
- **Training-loop ready** -- repeated forwards aggregate per layer (min/mean/max latency, forward count) instead of unbounded growth
- **Accurate latency** -- pre-hook to post-hook timing using `torch.cuda.Event` on GPU and `perf_counter` on CPU
- **Activation statistics** -- mean, std, min, max, NaN count, Inf count per tensor (gated by `capture_stats`)
- **Backward gradient tracing** -- grad norm, mean, NaN detection, zero-gradient detection
- **GPU memory tracking** -- per-layer forward & backward `torch.cuda.memory_allocated()` / `memory_reserved()` deltas (CUDA only)
- **Anomaly detection** -- 8 built-in checks: NaN, Inf, dead layers, exploding variance, variance spikes, zero gradients, NaN gradients, empty outputs
- **Configurable thresholds** -- `TraceSession(model, thresholds=Thresholds(...))` for tuning dead-layer, exploding-variance, and spike-ratio sensitivity
- **Order-independent variance spikes** -- per-layer input→output std ratio, not adjacent-layer order in execution trace, so branched / shared / control-flow models no longer false-flag
- **Trace diff / CI mode** -- `tracetorch diff baseline.json current.json` for regression checks with exit codes
- **Live watch mode** -- `TraceSession(model, watch_path=...)` flushes the trace to disk periodically during training; `tracetorch watch trace.json` re-renders a live terminal view on each write
- **JSON export** -- structured, flat format designed for dashboards and CI pipelines; safely serializes non-native metadata
- **Rich CLI inspector** -- nested layer tree, anomaly warnings, model summary in the terminal
- **Accurate latency** -- pre-hook to post-hook timing using `torch.cuda.Event` on GPU and `perf_counter` on CPU, not just stat-computation overhead

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
  <img src="assets/ss.jpg?v=4" alt="TraceTorch CLI output" width="100%">
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

Run backward inside the session to capture gradient data, including
**backward latency** (symmetric to forward latency) and **`grad_in`** (the
gradient arriving at the layer's inputs, useful for spotting broken gradient
paths):

```python
session = TraceSession(model)

with session:
    x = torch.randn(4, 10, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

for layer in session.record.layers:
    print(
        layer.full_name,
        "fwd:", layer.latency_ms_mean, "ms x", layer.forward_count,
        "bwd:", layer.bwd_latency_ms_mean, "ms x", layer.backward_count,
        "grad_out:", layer.grad_norm,
        "grad_in:", layer.grad_in_norm,
    )
```

Gradient fields stay `None` / `False` and `backward_count` is 0 if no
backward pass runs.

## GPU memory tracking

On CUDA, TraceTorch captures per-layer memory deltas around both the forward
and backward passes via `torch.cuda.memory_allocated()` /
`torch.cuda.memory_reserved()`. CPU traces leave these fields `None`.

```python
session = TraceSession(model)

with session:
    if torch.cuda.is_available():
        x = torch.randn(32, 10, device="cuda")
        model(x).sum().backward()

for layer in session.record.layers:
    print(
        layer.full_name,
        "fwd:", layer.fwd_mem_alloc_delta,
        "bwd:", layer.bwd_mem_alloc_delta,
    )
```

`session.summary()` surfaces the heaviest forward & backward allocators —
usually the OOM culprit.

## Training loops

Trace inside a training loop safely. Each module's forward is aggregated per
instance rather than appended as a duplicate, so memory stays bounded by the
number of *layers*, not the number of *steps*.

```python
session = TraceSession(model)

with session:
    for batch in dataloader:
        out = model(batch)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Aggregated per-layer latency is available on the trace:
for layer in session.record.layers:
    print(
        layer.full_name,
        layer.forward_count,
        layer.latency_ms_mean,
        layer.latency_ms_min,
        layer.latency_ms_max,
    )
```

If you only need shapes/latency and want to avoid the per-tensor stats cost,
disable stats:

```python
TraceSession(model, capture_stats=False)
```

A `TraceSession` can be re-entered: each `with` block starts from a clean
state, so layer traces and anomalies do not leak across runs.

## Live watch mode

For long training runs, TraceTorch can flush its in-progress trace to disk
periodically so you can observe layer latency, memory, and gradients as they
accumulate -- without waiting for the session to exit.

```python
session = TraceSession(
    model,
    watch_path="./trace.json",   # flush target
    watch_interval_s=2.0,         # seconds between flushes
)

with session:
    for epoch in range(10):
        for batch in dataloader:
            out = model(batch)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

While that runs, open a second terminal:

```bash
tracetorch watch ./trace.json --interval 0.5
```

The watch command polls the file's mtime and re-renders the model summary,
layer tree, and warnings on each change. Press `Ctrl+C` to stop watching.

The watch file reflects the **partial** state during training (no anomaly
analysis until `__exit__`). On exit, the session does a final flush that
includes anomalies, `total_time_ms`, and the complete aggregated latency.

Filters work on `watch` too:

```bash
tracetorch watch ./trace.json --type Linear --min-latency 0.5
```

## Trace diff (CI mode)

Compare two trace files to detect regressions between branches, commits, or
builds. Returns a non-zero exit code when regressions are present, suitable
for CI gates.

```bash
# human-readable
tracetorch diff baseline.json current.json

# machine-readable JSON (pipe into another tool)
tracetorch diff baseline.json current.json --json

# custom thresholds (relative delta, e.g. 0.1 = 10%)
tracetorch diff baseline.json current.json \
    --latency-threshold 0.1 \
    --grad-norm-threshold 0.25 \
    --std-threshold 0.25
```

Detected changes:

| Kind | Severity | Trigger |
| --- | --- | --- |
| `layer_added` / `layer_removed` | warning | Layer present in only one trace |
| `latency_regression` | warning | Mean latency increased beyond threshold |
| `latency_improvement` | info | Mean latency decreased beyond threshold |
| `grad_norm_change` | warning / info | Gradient norm delta beyond threshold |
| `activation_std_change` | warning / info | Output std delta beyond threshold |
| `shape_change` | critical | Output shape differs between traces |
| `new_anomaly` | warning / critical | Anomaly present in current but not baseline |
| `resolved_anomaly` | info | Anomaly present in baseline but not current |

Programmatic API:

```python
from tracetorch.storage import load_json
from tracetorch import diff_records

baseline = load_json("baseline.json")
current = load_json("current.json")
diff = diff_records(baseline, current, latency_threshold=0.1)

print(diff.has_regressions, diff.critical_count, diff.warning_count)
for entry in diff.entries:
    print(entry.severity, entry.kind, entry.layer, entry.message)
```

## Anomaly checks

| Type | Trigger |
| --- | --- |
| `nan_activation` | Output tensor contains NaN |
| `inf_activation` | Output tensor contains Inf |
| `dead_layer` | Output mean and std both near zero (configurable: `dead_layer_mean`, `dead_layer_std`) |
| `exploding_variance` | Std > `exploding_variance_std` (default 100), or output std > `variance_spike_ratio`× input std (default 10×) |
| `high_variance` | Std between `high_variance_min` (10) and `exploding_variance_std` (100) |
| `empty_output` | No tensor output captured |
| `zero_gradient` | Backward gradient is all zeros |
| `nan_gradient` | Backward gradient contains NaN |

The variance-spike check is **per-layer** (input std vs output std of the
same module) so it is independent of execution order; branched /
shared-submodule / control-flow models do not produce false positives from
adjacent-but-unrelated layers that happened to run earlier.

### Tuning thresholds

```python
from tracetorch import Thresholds

session = TraceSession(
    model,
    thresholds=Thresholds(
        dead_layer_mean=1e-4,
        dead_layer_std=1e-4,
        exploding_variance_std=200.0,
        high_variance_min=20.0,
        variance_spike_ratio=20.0,
    ),
)
```

Programmatic detection uses the same threshold plumbing:

```python
from tracetorch import Thresholds, detect_anomalies

anomalies = detect_anomalies(record, thresholds=Thresholds(dead_layer_std=1e-4))
```

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
      "latency_ms_min": 0.18,
      "latency_ms_max": 0.31,
      "latency_ms_mean": 0.25,
      "forward_count": 1,
"grad_norm": null,
      "grad_mean": null,
      "grad_has_nan": false,
      "has_zero_grad": false,
      "grad_in_norm": null,
      "grad_in_mean": null,
      "grad_in_has_nan": false,
      "bwd_latency_ms": 0.0,
      "bwd_latency_ms_min": null,
      "bwd_latency_ms_max": null,
      "bwd_latency_ms_mean": null,
      "backward_count": 0,
      "fwd_mem_alloc_delta": null,
      "fwd_mem_reserved_delta": null,
      "bwd_mem_alloc_delta": null,
      "bwd_mem_reserved_delta": null,
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
session = TraceSession(
    model,
    capture_stats=True,
    model_name=None,
    thresholds=None,
    watch_path=None,
    watch_interval_s=1.0,
)
```

Context manager for one traced execution window. Safe to re-enter; each
`with` block resets the captured record so layer traces and anomalies do not
leak between runs.

| Property / Method | Description |
| --- | --- |
| `record` | Collected `TraceRecord` |
| `anomalies` | `list[Anomaly]` populated after exit |
| `export(path)` | Write JSON, return `Path` |
| `summary()` | Compact text summary |

### diff_records

```python
from tracetorch import diff_records

diff = diff_records(baseline, current, latency_threshold=0.1,
                    grad_norm_threshold=0.25, std_threshold=0.25)
```

Returns a `TraceDiff` with grouped diff entries; `.has_regressions` is true
when any warning or critical entry is present.

### Data structures

```python
from tracetorch import TraceRecord, LayerTrace, TensorStats, Anomaly
from tracetorch import Thresholds, TraceDiff, DiffSeverity, diff_records
```

All dataclasses serialize via `.to_dict()`. JSON format is flat for dashboard consumption.

### CLI

```bash
tracetorch inspect trace.json         # rich inspection
tracetorch summary trace.json         # just the model summary table
tracetorch watch trace.json           # live re-render on file change
tracetorch diff a.json b.json         # CI regression diff (exit non-zero)
tracetorch diff a.json b.json --json  # machine-readable diff

# Filtering on `inspect` and `watch`:
tracetorch inspect trace.json --filter 'encoder.*'      # glob on full_name
tracetorch inspect trace.json --type Linear              # only Linear layers
tracetorch inspect trace.json --min-latency 1.0          # ms (uses mean latency)
tracetorch inspect trace.json --json                     # machine-readable

# Watch options:
tracetorch watch trace.json --interval 1.0               # poll every 1s
tracetorch watch trace.json --type Linear --min-latency 0.5
```

Filters can be combined; `--json` respects the filter so you can pipe a
subset of layers into another tool.

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
