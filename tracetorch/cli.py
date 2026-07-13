"""CLI interface for TraceTorch with rich terminal output."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from tracetorch.collector import LayerTrace, TraceRecord
from tracetorch.diff import DiffSeverity, TraceDiff, diff_records
from tracetorch.storage import load_json
from tracetorch.utils import format_bytes, format_ms, format_params, format_shape

console = Console()


@dataclass
class FilterOptions:
    """Optional filters applied to a trace before display.

    All fields are optional; ``None`` / empty means "no constraint".
    """

    name_pattern: str | None = None
    module_type: str | None = None
    min_latency_ms: float | None = None

    def matches(self, layer: LayerTrace) -> bool:
        if self.name_pattern is not None and not fnmatch.fnmatch(
            layer.full_name, self.name_pattern
        ):
            return False
        if self.module_type is not None and layer.module_type != self.module_type:
            return False
        if self.min_latency_ms is not None:
            lat = layer.latency_ms_mean if layer.latency_ms_mean else layer.latency_ms
            if lat < self.min_latency_ms:
                return False
        return True


def _filtered_layers(record: TraceRecord, opts: FilterOptions) -> list[LayerTrace]:
    return [layer for layer in record.layers if opts.matches(layer)]


def inspect_command(
    trace_path: str,
    *,
    filter_opts: FilterOptions | None = None,
    json_output: bool = False,
) -> None:
    """Display a rich inspection of a trace file."""
    path = Path(trace_path)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    record = load_json(path)
    opts = filter_opts if filter_opts is not None else FilterOptions()

    if json_output:
        out: dict[str, Any] = {
            "metadata": record.metadata,
            "layers": [layer.to_dict() for layer in _filtered_layers(record, opts)],
            "warnings": record.warnings,
        }
        sys.stdout.write(json.dumps(out, indent=2, ensure_ascii=False, default=str))
        sys.stdout.write("\n")
        return

    _print_model_summary(record)
    _print_layer_tree(record, opts)
    _print_warnings(record)


def summary_command(trace_path: str) -> None:
    """Print just the model summary table (no layer tree, no warnings)."""
    path = Path(trace_path)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    record = load_json(path)
    _print_model_summary(record)


def watch_command(
    trace_path: str,
    *,
    interval: float = 0.5,
    filter_opts: FilterOptions | None = None,
) -> None:
    """Watch a trace file and re-render the inspect view on change.

    Polls the file's mtime at ``interval`` seconds. When the file changes
    (e.g. a ``TraceSession`` with ``watch_path`` is flushing during training),
    the screen is cleared and the model summary + layer tree + warnings are
    re-rendered. Press Ctrl+C to exit.
    """
    path = Path(trace_path)
    if not path.exists():
        console.print(f"[yellow]Waiting for {path} to appear... (Ctrl+C to exit)[/yellow]")
        while not path.exists():
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                console.print("\n[dim]Stopped.[/dim]")
                return

    opts = filter_opts if filter_opts is not None else FilterOptions()
    last_mtime: float | None = None
    first_render = True

    try:
        while True:
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                mtime = None

            if mtime != last_mtime or first_render:
                last_mtime = mtime
                first_render = False
                console.clear()
                console.print(f"[dim]Watching {path} — Ctrl+C to exit[/dim]\n")
                if mtime is not None:
                    try:
                        record = load_json(path)
                    except (json.JSONDecodeError, OSError):
                        # File may be mid-write; skip this cycle.
                        console.print("[yellow]File changed — waiting for stable write...[/yellow]")
                    else:
                        _print_model_summary(record)
                        _print_layer_tree(record, opts)
                        _print_warnings(record)
                else:
                    console.print("[yellow]File disappeared — waiting...[/yellow]")

            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching.[/dim]")


def diff_command(
    baseline_path: str,
    current_path: str,
    *,
    latency_threshold: float,
    grad_norm_threshold: float,
    std_threshold: float,
    json_output: bool,
) -> None:
    """Diff two trace files and report regressions for CI use."""
    base_path = Path(baseline_path)
    curr_path = Path(current_path)
    for p in (base_path, curr_path):
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            sys.exit(1)

    baseline = load_json(base_path)
    current = load_json(curr_path)
    diff = diff_records(
        baseline,
        current,
        latency_threshold=latency_threshold,
        grad_norm_threshold=grad_norm_threshold,
        std_threshold=std_threshold,
    )

    if json_output:
        sys.stdout.write(json.dumps(diff.to_dict(), indent=2, ensure_ascii=False))
        sys.stdout.write("\n")
        sys.exit(1 if diff.has_regressions else 0)

    _print_diff_header(diff)
    _print_diff_entries(diff)
    _print_diff_summary(diff)

    # CI exit code: non-zero when regressions are present.
    if diff.has_regressions:
        sys.exit(1)


def _print_diff_header(diff: TraceDiff) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Baseline", diff.baseline_model)
    table.add_row("Current", diff.current_model)
    table.add_row("Critical", str(diff.critical_count))
    table.add_row("Warnings", str(diff.warning_count))
    console.print()
    console.print(
        Panel(table, title="[bold]Trace Diff[/bold]", border_style="magenta", expand=False)
    )


def _print_diff_entries(diff: TraceDiff) -> None:
    if not diff.entries:
        console.print("[green]No changes detected between traces.[/green]")
        return

    table = Table(title="Diff entries", border_style="magenta")
    table.add_column("Severity", style="bold", width=10)
    table.add_column("Kind", style="dim", width=22)
    table.add_column("Layer", style="cyan")
    table.add_column("Message")

    for e in diff.entries:
        if e.severity == DiffSeverity.CRITICAL:
            style, icon = "red", "✗"
        elif e.severity == DiffSeverity.WARNING:
            style, icon = "yellow", "⚠"
        else:
            style, icon = "dim", "ℹ"
        table.add_row(
            f"[{style}]{icon} {e.severity.value}[/{style}]",
            e.kind,
            e.layer,
            e.message,
        )

    console.print()
    console.print(table)


def _print_diff_summary(diff: TraceDiff) -> None:
    console.print()
    if diff.has_regressions:
        console.print(
            f"[red]✗ {diff.critical_count} critical, "
            f"{diff.warning_count} warning(s) — regressions detected.[/red]"
        )
    else:
        console.print("[green]✓ No regressions detected.[/green]")


def _print_model_summary(record: TraceRecord) -> None:
    meta = record.metadata
    total_params = sum(layer.params for layer in record.layers)
    total_ms = meta.get("total_time_ms", 0)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    total_forwards = max(
        (layer.forward_count for layer in record.layers),
        default=0,
    )

    table.add_row("Model", meta.get("model_name", "Unknown"))
    table.add_row("PyTorch", meta.get("pytorch_version", "?"))
    table.add_row("Layers", str(len(record.layers)))
    table.add_row("Forward passes", str(total_forwards))
    table.add_row("Parameters", format_params(total_params))
    table.add_row("Execution Time", format_ms(total_ms))

    console.print()
    console.print(
        Panel(table, title="[bold]Model Summary[/bold]", border_style="blue", expand=False)
    )


def _print_layer_tree(record: TraceRecord, opts: FilterOptions) -> None:
    tree = Tree("[bold]Layers[/bold]")

    # Group layers by nesting depth for tree structure
    stack: list[tuple[int, Tree]] = [(-1, tree)]

    layers = _filtered_layers(record, opts)
    if not layers:
        console.print()
        console.print("[dim]No layers matched the filter.[/dim]")
        return

    for layer in layers:
        depth = layer.depth
        label = _format_layer_label(layer)

        # Pop stack until we find the parent
        while len(stack) > 1 and stack[-1][0] >= depth:
            stack.pop()

        parent_tree = stack[-1][1]
        child = parent_tree.add(label)
        stack.append((depth, child))

    console.print()
    console.print(tree)


def _format_layer_label(layer: LayerTrace) -> str:
    parts: list[str] = []
    parts.append(f"[bold]{layer.name}[/bold]")
    parts.append(f"[dim]({layer.module_type})[/dim]")

    if layer.params > 0:
        parts.append(f"Params: {format_params(layer.params)}")

    if layer.forward_count > 1 and (layer.latency_ms_mean or 0) > 0:
        parts.append(
            f"Latency: {format_ms(layer.latency_ms_mean or 0)} mean "
            f"({format_ms(layer.latency_ms_min or 0)}-{format_ms(layer.latency_ms_max or 0)} "
            f"x{layer.forward_count})"
        )
    elif layer.latency_ms > 0:
        parts.append(f"Latency: {format_ms(layer.latency_ms)}")

    if layer.backward_count > 1 and (layer.bwd_latency_ms_mean or 0) > 0:
        parts.append(
            f"Bwd: {format_ms(layer.bwd_latency_ms_mean or 0)} mean "
            f"({format_ms(layer.bwd_latency_ms_min or 0)}-"
            f"{format_ms(layer.bwd_latency_ms_max or 0)} "
            f"x{layer.backward_count})"
        )
    elif layer.bwd_latency_ms > 0:
        parts.append(f"Bwd: {format_ms(layer.bwd_latency_ms)}")

    if layer.grad_in_norm is not None:
        parts.append(f"grad_in_norm: {layer.grad_in_norm:.4f}")

    # Output stats
    for out in layer.outputs:
        shape_str = format_shape(out.shape)
        parts.append(f"Output: {shape_str}")
        if out.stats and out.stats.std is not None:
            std_val = out.stats.std
            if std_val < 10:
                parts.append(f"std: {std_val:.2f} ✓")
            else:
                parts.append(f"std: {std_val:.2f} ⚠")

    if layer.has_nan:
        parts.append("[red]✗ NaN[/red]")
    if layer.has_inf:
        parts.append("[red]✗ Inf[/red]")

    # GPU memory deltas (CUDA only, None on CPU).
    if layer.fwd_mem_alloc_delta is not None:
        parts.append(f"fwd Δmem: {format_bytes(layer.fwd_mem_alloc_delta)}")
    if layer.bwd_mem_alloc_delta is not None:
        parts.append(f"bwd Δmem: {format_bytes(layer.bwd_mem_alloc_delta)}")

    return " │ ".join(parts)


def _print_warnings(record: TraceRecord) -> None:
    if not record.warnings:
        console.print()
        console.print("[green]✓ No anomalies detected[/green]")
        return

    table = Table(title="Warnings & Anomalies", border_style="yellow")
    table.add_column("Severity", style="bold", width=10)
    table.add_column("Layer", style="cyan")
    table.add_column("Message")

    for w in record.warnings:
        severity = w.get("severity", "info")
        if severity == "critical":
            style = "red"
            icon = "✗"
        elif severity == "warning":
            style = "yellow"
            icon = "⚠"
        else:
            style = "dim"
            icon = "ℹ"

        table.add_row(
            f"[{style}]{icon} {severity}[/{style}]",
            w.get("layer", "?"),
            w.get("message", ""),
        )

    console.print()
    console.print(table)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tracetorch",
        description="A forensic trace console for PyTorch models.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a trace file")
    inspect_parser.add_argument("trace_file", help="Path to trace JSON file")
    inspect_parser.add_argument(
        "--filter",
        dest="name_pattern",
        default=None,
        help="Glob pattern to filter layers by full_name (e.g. 'encoder.*.linear')",
    )
    inspect_parser.add_argument(
        "--type",
        dest="module_type",
        default=None,
        help="Only show layers of this module type (e.g. 'Linear')",
    )
    inspect_parser.add_argument(
        "--min-latency",
        dest="min_latency_ms",
        type=float,
        default=None,
        help="Only show layers with mean latency >= this value (ms)",
    )
    inspect_parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit machine-readable JSON instead of rich output",
    )

    summary_parser = subparsers.add_parser(
        "summary", help="Print just the model summary table for a trace file"
    )
    summary_parser.add_argument("trace_file", help="Path to trace JSON file")

    watch_parser = subparsers.add_parser(
        "watch", help="Watch a trace file and re-render on change (live training view)"
    )
    watch_parser.add_argument("trace_file", help="Path to trace JSON file to watch")
    watch_parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds (default: 0.5)",
    )
    watch_parser.add_argument(
        "--filter",
        dest="name_pattern",
        default=None,
        help="Glob pattern to filter layers by full_name",
    )
    watch_parser.add_argument(
        "--type",
        dest="module_type",
        default=None,
        help="Only show layers of this module type",
    )
    watch_parser.add_argument(
        "--min-latency",
        dest="min_latency_ms",
        type=float,
        default=None,
        help="Only show layers with mean latency >= this value (ms)",
    )

    diff_parser = subparsers.add_parser(
        "diff", help="Diff two trace files for CI regression checks"
    )
    diff_parser.add_argument("baseline", help="Path to baseline trace JSON file")
    diff_parser.add_argument("current", help="Path to current trace JSON file")
    diff_parser.add_argument(
        "--latency-threshold",
        type=float,
        default=0.1,
        help="Relative latency delta that flags a regression (default: 0.1 = 10%%)",
    )
    diff_parser.add_argument(
        "--grad-norm-threshold",
        type=float,
        default=0.25,
        help="Relative grad-norm delta that flags a change (default: 0.25)",
    )
    diff_parser.add_argument(
        "--std-threshold",
        type=float,
        default=0.25,
        help="Relative activation-std delta that flags a change (default: 0.25)",
    )
    diff_parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit machine-readable JSON instead of rich tables",
    )

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_command(
            args.trace_file,
            filter_opts=FilterOptions(
                name_pattern=args.name_pattern,
                module_type=args.module_type,
                min_latency_ms=args.min_latency_ms,
            ),
            json_output=args.json_output,
        )
    elif args.command == "summary":
        summary_command(args.trace_file)
    elif args.command == "watch":
        watch_command(
            args.trace_file,
            interval=args.interval,
            filter_opts=FilterOptions(
                name_pattern=args.name_pattern,
                module_type=args.module_type,
                min_latency_ms=args.min_latency_ms,
            ),
        )
    elif args.command == "diff":
        diff_command(
            args.baseline,
            args.current,
            latency_threshold=args.latency_threshold,
            grad_norm_threshold=args.grad_norm_threshold,
            std_threshold=args.std_threshold,
            json_output=args.json_output,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
