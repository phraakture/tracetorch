"""Helpers and formatters for TraceTorch."""

from __future__ import annotations

import math
from typing import Any

import torch


def compute_tensor_stats(
    tensor: torch.Tensor,
    *,
    sample_threshold: int = 1_000_000,
) -> dict[str, Any]:
    """Compute descriptive statistics for a tensor.

    Returns a dict with mean, std, min, max, nan_count, and inf_count.
    Non-floating or empty tensors return None for the float-only fields.

    Implementation notes:
    - NaN / Inf counts are computed over the **whole** tensor (not the
      sampled subset) since those are the primary anomaly signals and have
      cheap mask-sum cost.
    - Mean / std / min / max are computed on a strided subsample when the
      tensor exceeds ``sample_threshold`` elements so cost stays bounded
      for very large activations; the resulting statistics approximate the
      tensor's true moments but the strided sample is unbiased.
    - Uses ``aminmax()`` and ``std()`` fused kernels where possible.
    """
    with torch.no_grad():
        if not tensor.is_floating_point() or tensor.numel() == 0:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "nan_count": 0,
                "inf_count": 0,
            }

        # Full-tensor NaN / Inf counts: cheap and exact, used for anomaly
        # detection. Masks are kept for the subsampling pass below.
        nan_mask = torch.isnan(tensor)
        inf_mask = torch.isinf(tensor)
        nan_count = int(nan_mask.sum().item())
        inf_count = int(inf_mask.sum().item())

        # Optional subsample for the mean/std/min/max pass on huge tensors.
        for_stats = tensor
        if tensor.numel() > sample_threshold and tensor.numel() > 1:
            step = tensor.numel() // sample_threshold
            for_stats = tensor.reshape(-1)[::step]
            # Resample the masks to match; keep this consistent with the
            # subsample so the "clean" subset is computed against the right
            # finite-mask view.
            flat_nan = nan_mask.reshape(-1)[::step]
            flat_inf = inf_mask.reshape(-1)[::step]
        else:
            flat_nan = nan_mask
            flat_inf = inf_mask

        # Use float view only when needed for std/mean precision on half types.
        if for_stats.dtype not in (torch.float32, torch.float64):
            for_stats = for_stats.float()

        clean = for_stats[~(flat_nan | flat_inf)]
        if clean.numel() == 0:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "nan_count": nan_count,
                "inf_count": inf_count,
            }

        cmin, cmax = clean.aminmax()
        cmean = clean.mean()
        if clean.numel() > 1:
            cstd = clean.std(unbiased=False)
        else:
            cstd = torch.zeros((), dtype=clean.dtype)

        return {
            "mean": float(cmean.item()),
            "std": float(cstd.item()),
            "min": float(cmin.item()),
            "max": float(cmax.item()),
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


def format_bytes(num_bytes: float | None) -> str:
    """Format a byte count in human-readable form (B/KiB/MiB/GiB).

    Negative values are supported and used to show memory-release deltas.
    ``None`` is rendered as ``-`` so callers can display "CPU layers" cleanly.
    """
    if num_bytes is None:
        return "-"
    sign = "-" if num_bytes < 0 else ""
    n = abs(num_bytes)
    if n >= 1024**3:
        return f"{sign}{n / 1024**3:.2f}GiB"
    if n >= 1024**2:
        return f"{sign}{n / 1024**2:.2f}MiB"
    if n >= 1024:
        return f"{sign}{n / 1024:.1f}KiB"
    return f"{sign}{n:.0f}B"


def format_shape(shape: list[int] | tuple[int, ...]) -> str:
    """Format tensor shape as a readable string."""
    return "[" + ", ".join(str(d) for d in shape) + "]"


def format_ms(ms: float) -> str:
    """Format a millisecond value with appropriate precision."""
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
