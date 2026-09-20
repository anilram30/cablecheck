"""
The end-to-end pipeline: files in, verdict + report + database row out.

    read Touchstone  ->  fixture removal  ->  mixed-mode quantities
    ->  limit comparison  ->  Evaluation  ->  HTML/PDF/JSON report + SQLite row

A *measurement set* is one or more files, each with its own port map
(e.g. a 4-port file per pair, plus a 4-port near-end crosstalk file across
two pairs, or a single 8-port file), all belonging to one physical sample.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .deembed import bisect_2x_thru_pair, deembed, port_extension
from .evaluate import Evaluation, evaluate
from .io import read_touchstone
from .limits.library import CableType, load_cable_type
from .mixedmode import PortMap
from .network import Network
from .quantities import Trace, compute_quantities

__all__ = ["SampleInfo", "MeasurementFile", "FixtureSpec", "RunResult", "run_sample", "sha256_file"]


@dataclass
class SampleInfo:
    sample_id: str
    cable_type: str
    length_m: float | None = None
    lot: str = ""
    part_number: str = ""
    operator: str = ""
    instrument: str = ""
    instrument_serial: str = ""
    calibration_date: str = ""
    temperature_c: float | None = None
    humidity_pct: float | None = None
    site: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FixtureSpec:
    """How to remove the fixture from one file."""
    method: str = "none"                 # none | files | 2xthru | port-extension
    left: str | None = None              # Touchstone path (VNA side ports first, DUT side last)
    right: str | None = None
    thru: str | None = None              # 2x-thru Touchstone (4-port for a pair)
    delays_ps: list[float] | None = None  # port extension
    loss_db_at_1ghz: list[float] | None = None


@dataclass
class MeasurementFile:
    path: str
    port_map: str = "A+near,A-near,A+far,A-far"
    fixture: FixtureSpec = field(default_factory=FixtureSpec)
    quantities: list[str] | None = None  # restrict what this file contributes


@dataclass
class RunResult:
    sample: SampleInfo
    cable_type: CableType
    evaluation: Evaluation
    traces: list[Trace]
    networks: dict[str, Network]         # de-embedded networks per file
    raw_networks: dict[str, Network]
    files: list[dict]                    # path, sha256, nports, fmin, fmax, npoints, fixture method
    software_version: str = __version__
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    warnings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return self.evaluation.verdict

    def to_json_dict(self, include_traces: bool = True) -> dict:
        d = {
            "software": {"name": "cablecheck", "version": self.software_version},
            "timestamp": self.timestamp,
            "sample": self.sample.to_dict(),
            "cable_type": {"id": self.cable_type.id, "title": self.cable_type.title,
                           "standard": self.cable_type.standard, "status": self.cable_type.status,
                           "file": self.cable_type.path},
            "files": self.files,
            "verdict": self.verdict,
            "headline": self.evaluation.headline.to_dict() if self.evaluation.headline else None,
            "results": self.evaluation.summary_rows(),
            "warnings": self.warnings + self.evaluation.warnings,
        }
        if include_traces:
            d["traces"] = [{"quantity": r.quantity, "pair": r.pair, "unit": r.unit, "xunit": r.xunit,
                            "x": _round_list(r.x), "value": _round_list(r.value),
                            "limit": _round_list(r.limit) if r.limit is not None else None,
                            "limit_upper": _round_list(r.limit_upper) if r.limit_upper is not None else None}
                           for r in self.evaluation.results]
        return d


def _round_list(a, nd=6):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(a, float)]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _apply_fixture(net: Network, fx: FixtureSpec, warnings: list[str]) -> tuple[Network, str]:
    m = (fx.method or "none").lower()
    if m == "none":
        return net, "none"
    if m == "files":
        left = read_touchstone(fx.left) if fx.left else None
        right = read_touchstone(fx.right) if fx.right else left
        if left is None and right is None:
            raise ValueError("fixture method 'files' needs at least a left fixture file")
        return deembed(net, left, right), "files"
    if m in ("2xthru", "2x-thru", "thru"):
        if not fx.thru:
            raise ValueError("fixture method '2xthru' needs the 2x-thru Touchstone file")
        thru = read_touchstone(fx.thru)
        if thru.nports == 4:
            half = bisect_2x_thru_pair(thru)
        else:
            from .deembed import bisect_2x_thru
            half = bisect_2x_thru(thru)
        if half.nports != net.nports:
            raise ValueError(f"2x-thru has {thru.nports} ports but the measurement has {net.nports}")
        warnings.append(f"fixture from 2x-thru bisection: {half.comments[0]}")
        return deembed(net, half, half), "2xthru"
    if m in ("port-extension", "portext"):
        d = np.asarray(fx.delays_ps or [0.0], float) * 1e-12
        return port_extension(net, d, fx.loss_db_at_1ghz), "port-extension"
    raise ValueError(f"unknown fixture method {fx.method!r}")


def run_sample(sample: SampleInfo, files: list[MeasurementFile], cable_type: CableType | str | None = None,
               extra_limit_dirs: list[str] | None = None, nvp: float | None = None,
               profile_engine: str = "plain") -> RunResult:
    if cable_type is None:
        cable_type = sample.cable_type
    ct = cable_type if isinstance(cable_type, CableType) else load_cable_type(cable_type, extra_limit_dirs)
    warnings: list[str] = []
    traces: list[Trace] = []
    nets: dict[str, Network] = {}
    raws: dict[str, Network] = {}
    file_records = []
    for mf in files:
        raw = read_touchstone(mf.path)
        pm = PortMap.parse(mf.port_map)
        if len(pm) != raw.nports:
            raise ValueError(f"{mf.path}: port map has {len(pm)} entries but the file has {raw.nports} ports")
        pv = raw.passivity_violation().max()
        if pv > 0.02:
            warnings.append(f"{Path(mf.path).name}: raw data is non-passive by {pv:.3f} (max singular value - 1); "
                            "check calibration")
        net, used = _apply_fixture(raw, mf.fixture, warnings)
        raws[mf.path] = raw
        nets[mf.path] = net
        traces.extend(compute_quantities(net, pm, length_m=sample.length_m, t_rise=ct.rise_time_ps * 1e-12,
                                         nvp=nvp, connector_mask_m=ct.connector_mask_m,
                                         impedance_window_m=ct.impedance_window_m, want=mf.quantities,
                                         profile_engine=profile_engine))
        file_records.append({"path": str(mf.path), "name": Path(mf.path).name, "sha256": sha256_file(mf.path),
                             "nports": raw.nports, "fmin_hz": float(raw.f[0]), "fmax_hz": float(raw.f[-1]),
                             "npoints": int(raw.nfreq), "port_map": mf.port_map, "fixture": used})
    # a quantity delivered by several files for the same pair keeps the first one
    seen = set()
    unique = []
    for t in traces:
        key = (t.quantity, t.pair)
        if key in seen:
            warnings.append(f"duplicate quantity {t.quantity}[{t.pair}] from a second file was ignored")
            continue
        seen.add(key)
        unique.append(t)
    ev = evaluate(unique, ct, sample.length_m)
    return RunResult(sample, ct, ev, unique, nets, raws, file_records, warnings=warnings)


def write_json(result: RunResult, path: str | Path, include_traces: bool = True) -> Path:
    path = Path(path)
    path.write_text(json.dumps(result.to_json_dict(include_traces), indent=1), encoding="utf-8")
    return path
