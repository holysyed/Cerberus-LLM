"""Case-level classification metrics and linearly interpolated latency percentiles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor


@dataclass(frozen=True)
class Observation:
    """One labeled case. None means a required measurement was unavailable."""

    positive: bool
    predicted: bool | None
    elapsed_ms: float
    source: str


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Interpolate between adjacent order statistics, including singleton samples."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between zero and one")
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = floor(position)
    return ordered[lower] + (ordered[ceil(position)] - ordered[lower]) * (position - lower)


def summarize(observations: Sequence[Observation]) -> dict[str, int | float | None]:
    """Report coverage, confusion matrix, available-case scores and warm latencies."""
    valid = [o for o in observations if o.predicted is not None]
    tp = sum(o.predicted is True and o.positive for o in valid)
    fp = sum(o.predicted is True and not o.positive for o in valid)
    fn = sum(o.predicted is False and o.positive for o in valid)
    tn = sum(o.predicted is False and not o.positive for o in valid)
    elapsed = [o.elapsed_ms for o in valid]
    return {
        "cases": len(observations),
        "scored": len(valid),
        "unavailable": len(observations) - len(valid),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "p50_ms": percentile(elapsed, 0.5),
        "p95_ms": percentile(elapsed, 0.95),
    }
