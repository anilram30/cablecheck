"""
Compare computed quantities against a cable type's limit lines and produce
the verdict.

For every (quantity, pair) trace with a limit the *margin* is defined so that
a positive number always means "inside the limit":

    kind = max   : margin(f) = limit(f) - value(f)
    kind = min   : margin(f) = value(f) - limit(f)
    kind = range : margin    = min(value - lower, upper - value)

evaluated on the measured frequency points that fall inside the limit's
frequency span (limits are interpolated onto the measured grid, never the
other way round, so no measured point is lost).  The quantity passes when
min_f margin(f) >= 0, and the report carries the worst margin together with
the frequency (or distance, or nothing for a scalar) at which it occurs,
the measured value and the limit there, and how many points fail.

The overall verdict is PASS only if every limited quantity passed; the
*headline* is the quantity with the smallest margin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .limits.library import CableType, Limit
from .quantities import Trace

__all__ = ["QuantityResult", "Evaluation", "evaluate"]


@dataclass
class QuantityResult:
    quantity: str
    pair: str
    unit: str
    xunit: str
    kind: str | None                     # None = informational (no limit)
    x: np.ndarray
    value: np.ndarray
    limit: np.ndarray | None             # for range kind: lower bound
    limit_upper: np.ndarray | None
    margin: np.ndarray | None
    mask: np.ndarray                     # points inside the limit span
    passed: bool | None
    worst_margin: float | None
    x_worst: float | None
    value_at_worst: float | None
    limit_at_worst: float | None
    n_points: int
    n_fail: int
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.quantity}[{self.pair}]"

    @property
    def is_scalar(self) -> bool:
        return self.xunit == ""

    def to_dict(self) -> dict:
        return {
            "quantity": self.quantity, "pair": self.pair, "unit": self.unit, "xunit": self.xunit,
            "kind": self.kind, "passed": self.passed,
            "worst_margin": _f(self.worst_margin), "x_worst": _f(self.x_worst),
            "value_at_worst": _f(self.value_at_worst), "limit_at_worst": _f(self.limit_at_worst),
            "n_points": self.n_points, "n_fail": self.n_fail, "note": self.note,
        }


def _f(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


@dataclass
class Evaluation:
    cable_type: CableType
    results: list[QuantityResult]
    warnings: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def limited(self) -> list[QuantityResult]:
        return [r for r in self.results if r.kind is not None]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.limited) and bool(self.limited)

    @property
    def verdict(self) -> str:
        if not self.limited:
            return "NO LIMITS"
        return "PASS" if self.passed else "FAIL"

    @property
    def headline(self) -> QuantityResult | None:
        lim = [r for r in self.limited if r.worst_margin is not None]
        return min(lim, key=lambda r: r.worst_margin) if lim else None

    def summary_rows(self) -> list[dict]:
        return [r.to_dict() for r in self.results]


def _eval_one(tr: Trace, lim: Limit | None, length_m: float | None) -> QuantityResult:
    x, y = tr.x, tr.y
    if lim is None:
        return QuantityResult(tr.quantity, tr.pair, tr.unit, tr.xunit, None, x, y, None, None, None,
                              np.ones_like(y, dtype=bool), None, None, None, None, None, y.size, 0,
                              note="informational (no limit for this cable type)")
    if tr.is_scalar:
        f_eval = np.array([1e6])  # any frequency inside the scalar limit's dummy span
    else:
        f_eval = x if tr.xunit == "Hz" else np.array([1e6])
    limval, mask = lim.evaluate(f_eval, length_m)
    if tr.is_scalar or tr.xunit != "Hz":
        # scalar or distance-based: broadcast the single limit value
        mask = np.ones_like(y, dtype=bool)
        if lim.kind == "range":
            lo, hi = np.full(y.shape, limval[0][0]), np.full(y.shape, limval[1][0])
        else:
            lo, hi = np.full(y.shape, limval[0]), None
    else:
        if lim.kind == "range":
            lo, hi = limval
        else:
            lo, hi = limval, None
    if lim.kind == "max":
        margin = lo - y
    elif lim.kind == "min":
        margin = y - lo
    elif lim.kind == "range":
        margin = np.minimum(y - lo, hi - y)
    else:
        raise ValueError(f"unknown limit kind {lim.kind}")
    margin = np.where(mask, margin, np.nan)
    if not np.any(mask & np.isfinite(y)):
        return QuantityResult(tr.quantity, tr.pair, tr.unit, tr.xunit, lim.kind, x, y, lo, hi, margin, mask,
                              None, None, None, None, None, 0, 0,
                              note="no measured points inside the limit's frequency span")
    valid = mask & np.isfinite(margin)
    k = int(np.nanargmin(np.where(valid, margin, np.inf)))
    n_fail = int(np.sum(valid & (margin < 0)))
    lim_at = float(lo[k]) if lim.kind != "range" else float(lo[k] if y[k] - lo[k] < hi[k] - y[k] else hi[k])
    return QuantityResult(tr.quantity, tr.pair, tr.unit, tr.xunit, lim.kind, x, y, lo, hi, margin, mask,
                          bool(n_fail == 0), float(margin[k]), float(x[k]) if np.isfinite(x[k]) else None,
                          float(y[k]), lim_at, int(valid.sum()), n_fail, note=lim.note)


def evaluate(traces: list[Trace], cable_type: CableType, length_m: float | None = None) -> Evaluation:
    results = []
    warnings = []
    if cable_type.max_length_m and length_m and length_m > cable_type.max_length_m + 1e-9:
        warnings.append(f"sample length {length_m} m exceeds the cable type's maximum "
                        f"{cable_type.max_length_m} m; the limits assume the shorter length")
    seen = set()
    for tr in traces:
        lim = cable_type.limit_for(tr.quantity)
        if lim is not None and lim.per_metre and not length_m:
            warnings.append(f"{tr.quantity}: limit is per metre but no sample length was given; treated as informational")
            lim = None
        results.append(_eval_one(tr, lim, length_m))
        seen.add(tr.quantity)
    for lim in cable_type.limits:
        if lim.quantity not in seen:
            warnings.append(f"limit '{lim.quantity}' defined for {cable_type.id} but the measurement "
                            "set does not provide that quantity (check the port map / missing files)")
    # order: limited quantities first, worst margin first
    results.sort(key=lambda r: (r.kind is None, r.worst_margin if r.worst_margin is not None else np.inf))
    return Evaluation(cable_type, results, warnings)
