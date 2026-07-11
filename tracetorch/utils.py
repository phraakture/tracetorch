"""Helpers and formatters for TraceTorch."""

from __future__ import annotations

import math
from typing import Any

import torch


def compute_tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    """Compute descriptive statistics for a tensor.

    Returns a dict with mean, std, min, max, nan_count, and inf_count.
    Handles non-floating tensors by returning None for float-only stats.
    """
    with torch.no_grad():
        is_floating = tensor.is_floating_point()

        if not is_floating or tensor.numel() == 0:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "nan_count": 0,
                "inf_count": 0,
            }

        flat = tensor.flatten().float()
        nan_count = int(torch.isnan(flat).sum().item())
        inf_count = int(torch.isinf(flat).sum().item())

        # Replace NaN/Inf for stat computation
        clean = flat[torch.isfinite(flat)]
        if clean.numel() == 0:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "nan_count": nan_count,
                "inf_count": inf_count,
            }

        return {
            "mean": float(clean.mean().item()),
            "std": float(clean.std().item()) if clean.numel() > 1 else 0.0,
            "min": float(clean.min().item()),
            "max": float(clean.max().item()),
            "nan_count": nan_count,
            "inf_count": inf_count,
        }


def format_params(count: int) -> str:
    """Format parameter count in human-readable form."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def format_shape(shape: list[int] | tuple[int, ...]) -> str:
    """Format tensor shape as a readable string."""
    return "[" + ", ".join(str(d) for d in shape) + "]"


def format_ms(seconds: float) -> str:
    """Convert seconds to milliseconds with appropriate precision."""
    ms = seconds * 1000
    if ms >= 100:
        return f"{ms:.0f}ms"
    if ms >= 10:
        return f"{ms:.1f}ms"
    if ms >= 1:
        return f"{ms:.2f}ms"
    return f"{ms:.3f}ms"


def is_nan(value: float | None) -> bool:
    """Check if a value is NaN."""
    return value is not None and math.isnan(value)


def is_inf(value: float | None) -> bool:
    """Check if a value is infinite."""
    return value is not None and math.isinf(value)
