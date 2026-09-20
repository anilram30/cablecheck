"""
Limit-line library.

A *cable type* is a TOML file with one ``[[limit]]`` table per quantity.
Each limit is a list of frequency segments; inside a segment the limit is
either a closed-form expression in ``f`` (MHz) or a table of points that is
interpolated linearly in log10(f).  Example::

    [meta]
    id = "1000base-t1-link-segment"
    title = "IEEE 802.3bp 1000BASE-T1 link segment (Type A)"
    ...

    [[limit]]
    quantity = "insertion_loss"
    kind = "max"                     # value must be <= limit
    unit = "dB"
    [[limit.segment]]
    fmin_mhz = 1
    fmax_mhz = 600
    expr = "0.0023*f + 0.5907*sqrt(f) + 0.0639/sqrt(f)"

Kinds: ``max`` (value <= limit passes), ``min`` (value >= limit passes),
``range`` (``lower <= value <= upper``; for scalars such as the mean
impedance).  A limit may carry ``per_metre = true`` to be multiplied by the
sample length, and ``scalar = true`` when it applies to a single number
rather than a curve.  ``provenance`` and ``status`` say where the numbers
came from and whether they have been checked against the controlled copy
of the standard - the lab is expected to keep its own files here.

Expressions are evaluated by a small AST walker that allows only
arithmetic, ``f``, ``length`` and the functions sqrt/log10/log/exp/min/max.
"""
from __future__ import annotations

import ast
import math
import operator
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

__all__ = ["Limit", "CableType", "load_cable_type", "list_cable_types", "safe_eval", "LimitError"]


class LimitError(ValueError):
    pass


# ------------------------------------------------------------- expression
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": np.sqrt, "log10": np.log10, "log": np.log, "exp": np.exp,
          "min": np.minimum, "max": np.maximum, "abs": np.abs}


def safe_eval(expr: str, **names):
    """Evaluate an arithmetic expression on numpy arrays with a fixed whitelist."""
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            if node.id == "pi":
                return math.pi
            raise LimitError(f"unknown name {node.id!r} in limit expression {expr!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise LimitError(f"disallowed syntax in limit expression {expr!r}")

    return ev(tree)


# ------------------------------------------------------------------ model
@dataclass
class Segment:
    fmin_mhz: float
    fmax_mhz: float
    expr: str | None = None
    points: list[list[float]] | None = None   # [[f_mhz, value], ...]
    lower_expr: str | None = None
    upper_expr: str | None = None

    def evaluate(self, f_mhz: np.ndarray, length: float | None) -> np.ndarray:
        names = {"f": f_mhz, "length": length if length is not None else np.nan}
        if self.expr is not None:
            return np.broadcast_to(safe_eval(self.expr, **names), f_mhz.shape).astype(float)
        if self.points is not None:
            p = np.array(self.points, float)
            return np.interp(np.log10(f_mhz), np.log10(p[:, 0]), p[:, 1])
        raise LimitError("segment needs 'expr' or 'points'")

    def evaluate_range(self, f_mhz: np.ndarray, length: float | None):
        names = {"f": f_mhz, "length": length if length is not None else np.nan}
        lo = np.broadcast_to(safe_eval(self.lower_expr, **names), f_mhz.shape).astype(float)
        hi = np.broadcast_to(safe_eval(self.upper_expr, **names), f_mhz.shape).astype(float)
        return lo, hi


@dataclass
class Limit:
    quantity: str
    kind: str                  # "max" | "min" | "range"
    unit: str = "dB"
    per_metre: bool = False
    scalar: bool = False
    segments: list[Segment] = field(default_factory=list)
    applies_to: str = "pair"   # "pair" | "pair-pair" | "all"
    note: str = ""

    @property
    def fmin_mhz(self) -> float:
        return min(s.fmin_mhz for s in self.segments) if self.segments else 0.0

    @property
    def fmax_mhz(self) -> float:
        return max(s.fmax_mhz for s in self.segments) if self.segments else np.inf

    def evaluate(self, f_hz: np.ndarray, length_m: float | None = None):
        """Return (limit array, mask of frequencies covered).  For ``range``
        limits the first element is a (lower, upper) tuple."""
        f_mhz = np.asarray(f_hz, float) / 1e6
        scale = 1.0
        if self.per_metre:
            if not length_m:
                raise LimitError(f"limit {self.quantity} is per metre but no sample length given")
            scale = float(length_m)
        if self.kind == "range":
            lo = np.full(f_mhz.shape, np.nan)
            hi = np.full(f_mhz.shape, np.nan)
        else:
            val = np.full(f_mhz.shape, np.nan)
        mask = np.zeros(f_mhz.shape, dtype=bool)
        for seg in self.segments:
            m = (f_mhz >= seg.fmin_mhz - 1e-9) & (f_mhz <= seg.fmax_mhz + 1e-9) & ~mask
            if not np.any(m):
                continue
            if self.kind == "range":
                l, h = seg.evaluate_range(f_mhz[m], length_m)
                lo[m], hi[m] = l * scale, h * scale
            else:
                val[m] = seg.evaluate(f_mhz[m], length_m) * scale
            mask |= m
        if self.kind == "range":
            return (lo, hi), mask
        return val, mask


@dataclass
class CableType:
    id: str
    title: str
    standard: str = ""
    clause: str = ""
    status: str = ""
    provenance: str = ""
    nominal_impedance: float = 100.0
    rise_time_ps: float = 500.0
    connector_mask_m: float = 0.5
    impedance_window_m: float = 3.0
    max_length_m: float | None = None
    limits: list[Limit] = field(default_factory=list)
    path: str = ""

    def limit_for(self, quantity: str) -> Limit | None:
        for l in self.limits:
            if l.quantity == quantity:
                return l
        return None

    @property
    def quantities(self) -> list[str]:
        return [l.quantity for l in self.limits]


# ---------------------------------------------------------------- loading
def _data_dir() -> Path:
    return Path(str(resources.files("cablecheck.limits") / "data"))


def list_cable_types(extra_dirs: list[str | Path] | None = None) -> list[CableType]:
    dirs = [_data_dir()] + [Path(d) for d in (extra_dirs or [])]
    out = []
    for d in dirs:
        if d.is_dir():
            for p in sorted(d.glob("*.toml")):
                out.append(load_cable_type(p))
    return out


def load_cable_type(ident: str | Path, extra_dirs: list[str | Path] | None = None) -> CableType:
    """Load by path, or by ``id`` searched in the built-in and ``extra_dirs`` folders."""
    p = Path(ident)
    if not p.is_file():
        dirs = [_data_dir()] + [Path(d) for d in (extra_dirs or [])]
        found = None
        for d in dirs:
            cand = d / f"{ident}.toml"
            if cand.is_file():
                found = cand
                break
            for q in d.glob("*.toml") if d.is_dir() else []:
                try:
                    if tomllib.loads(q.read_text())["meta"]["id"] == str(ident):
                        found = q
                        break
                except Exception:  # pragma: no cover - malformed foreign file
                    continue
            if found:
                break
        if found is None:
            raise LimitError(f"cable type {ident!r} not found (searched {[str(d) for d in dirs]})")
        p = found
    doc = tomllib.loads(p.read_text(encoding="utf-8"))
    meta = doc.get("meta", {})
    limits = []
    for l in doc.get("limit", []):
        segs = [Segment(**{k: v for k, v in s.items()}) for s in l.get("segment", [])]
        limits.append(Limit(quantity=l["quantity"], kind=l.get("kind", "max"), unit=l.get("unit", "dB"),
                            per_metre=l.get("per_metre", False), scalar=l.get("scalar", False),
                            segments=segs, applies_to=l.get("applies_to", "pair"), note=l.get("note", "")))
    return CableType(id=meta.get("id", p.stem), title=meta.get("title", p.stem),
                     standard=meta.get("standard", ""), clause=meta.get("clause", ""),
                     status=meta.get("status", ""), provenance=meta.get("provenance", ""),
                     nominal_impedance=meta.get("nominal_impedance", 100.0),
                     rise_time_ps=meta.get("rise_time_ps", 500.0),
                     connector_mask_m=meta.get("connector_mask_m", 0.5),
                     impedance_window_m=meta.get("impedance_window_m", 3.0),
                     max_length_m=meta.get("max_length_m"), limits=limits, path=str(p))
