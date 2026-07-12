"""CLI interface for TraceTorch with rich terminal output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from tracetorch.collector import LayerTrace, TraceRecord
from tracetorch.storage import load_json
from tracetorch.utils import format_ms, format_params, format_shape

console = Console()


def inspect_command(trace_path: str) -> None:
    """Display a rich inspection of a trace file."""
    path = Path(trace_path)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    record = load_json(path)
    _print_model_summary(record)
    _print_layer_tree(record)
    _print_warnings(record)


def _print_model_summary(record: TraceRecord) -> None:
    meta = record.metadata
    total_params = sum(layer.params for layer in record.layers)
    total_ms = meta.get("total_time_ms", 0)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Model", meta.get("model_name", "Unknown"))
    table.add_row("PyTorch", meta.get("pytorch_version", "?"))
    table.add_row("Layers", str(len(record.layers)))
    table.add_row("Parameters", format_params(total_params))
    table.add_row("Execution Time", format_ms(total_ms))

    console.print()
    console.print(
        Panel(table, title="[bold]Model Summary[/bold]", border_style="blue", expand=False)
    )


def _print_layer_tree(record: TraceRecord) -> None:
    tree = Tree("[bold]Layers[/bold]")

    # Group layers by nesting depth for tree structure
    stack: list[tuple[int, Tree]] = [(-1, tree)]

    for layer in record.layers:
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

    if layer.latency_ms > 0:
        parts.append(f"Latency: {format_ms(layer.latency_ms)}")

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

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_command(args.trace_file)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
