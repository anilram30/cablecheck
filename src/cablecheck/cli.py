"""Command-line entry point: ``cablecheck <subcommand> ...``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _default_port_map(nports: int) -> str:
    return {2: "se:near,se:far", 4: "A+near,A-near,A+far,A-far",
            8: "A+near,A-near,A+far,A-far,B+near,B-near,B+far,B-far"}.get(nports, "")


def _build_files(args) -> list:
    from .io import read_touchstone
    from .pipeline import FixtureSpec, MeasurementFile
    files = []
    pmaps = list(args.port_map or [])
    for i, p in enumerate(args.files):
        if i < len(pmaps):
            pm = pmaps[i]
        else:
            n = read_touchstone(p).nports
            pm = _default_port_map(n)
            if not pm:
                sys.exit(f"{p}: no default port map for {n} ports; pass --port-map")
        fx = FixtureSpec()
        if args.fixture_left or args.fixture_right:
            fx = FixtureSpec("files", left=args.fixture_left, right=args.fixture_right or args.fixture_left)
        elif args.thru:
            fx = FixtureSpec("2xthru", thru=args.thru)
        elif args.port_extension_ps is not None:
            fx = FixtureSpec("port-extension", delays_ps=[args.port_extension_ps])
        files.append(MeasurementFile(p, pm, fx))
    return files


def _write_outputs(result, out_dir: Path, want_pdf: bool, want_html: bool, want_json: bool, db_path: str | None):
    from .pipeline import write_json
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.sample.sample_id}_{result.timestamp.replace(':', '').replace('-', '')[:15]}"
    paths = {}
    if want_html:
        from .report.html import write_html
        paths["html"] = write_html(result, out_dir / f"{stem}.html")
    if want_pdf:
        from .report.pdf import write_pdf
        paths["pdf"] = write_pdf(result, out_dir / f"{stem}.pdf")
    if want_json:
        paths["json"] = write_json(result, out_dir / f"{stem}.json")
    if db_path:
        from .db import ResultsDB
        with ResultsDB(db_path) as db:
            paths["db_run_id"] = db.insert_run(result, str(paths.get("html") or paths.get("pdf") or ""))
    return paths


def _print_summary(result):
    from .report.plots import TITLES
    ev = result.evaluation
    print(f"sample {result.sample.sample_id}  cable type {result.cable_type.id}  ->  {ev.verdict}")
    h = ev.headline
    if h:
        at = f"{h.x_worst / 1e6:.2f} MHz" if h.xunit == "Hz" and h.x_worst else (f"{h.x_worst:.2f} m" if h.x_worst else "")
        print(f"  headline: {TITLES.get(h.quantity, h.quantity)} [{h.pair}] worst margin {h.worst_margin:+.2f} {h.unit} at {at}")
    for r in ev.results:
        word = "info" if r.passed is None else ("PASS" if r.passed else "FAIL")
        at = "" if r.x_worst is None else (f"@ {r.x_worst / 1e6:.2f} MHz" if r.xunit == "Hz" else f"@ {r.x_worst:.2f} m")
        m = "" if r.worst_margin is None else f"{r.worst_margin:+7.2f} {r.unit}"
        print(f"  {word:4s} {r.quantity:18s} {r.pair:14s} {m:14s} {at}")
    for w in result.warnings + ev.warnings:
        print(f"  warning: {w}")


def cmd_evaluate(args):
    from .pipeline import SampleInfo, run_sample
    sample = SampleInfo(sample_id=args.sample_id, cable_type=args.cable_type, length_m=args.length,
                        lot=args.lot or "", part_number=args.part_number or "", operator=args.operator or "",
                        instrument=args.instrument or "", instrument_serial=args.instrument_serial or "",
                        calibration_date=args.calibration_date or "", temperature_c=args.temperature,
                        site=args.site or "", notes=args.notes or "")
    result = run_sample(sample, _build_files(args), extra_limit_dirs=args.limit_dir, nvp=args.nvp,
                        profile_engine=args.profile_engine)
    _print_summary(result)
    paths = _write_outputs(result, Path(args.out), args.pdf, not args.no_html, args.json, args.db)
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0 if result.evaluation.passed else 1


def cmd_batch(args):
    """Job file: {"db": "...", "out": "...", "samples": [{"sample": {...SampleInfo...},
    "files": [{"path": ..., "port_map": ..., "fixture": {...}}]}]}"""
    from .pipeline import FixtureSpec, MeasurementFile, SampleInfo, run_sample
    job = json.loads(Path(args.job).read_text())
    out = Path(args.out or job.get("out", "reports"))
    db = args.db or job.get("db")
    n_fail = 0
    for entry in job["samples"]:
        sample = SampleInfo(**entry["sample"])
        files = [MeasurementFile(f["path"], f.get("port_map", _default_port_map(4)), FixtureSpec(**f.get("fixture", {})),
                                 f.get("quantities")) for f in entry["files"]]
        try:
            result = run_sample(sample, files, extra_limit_dirs=job.get("limit_dirs"))
        except Exception as e:  # keep the batch going
            print(f"sample {sample.sample_id}: ERROR {e}")
            n_fail += 1
            continue
        _print_summary(result)
        _write_outputs(result, out, args.pdf, True, True, db)
        n_fail += 0 if result.evaluation.passed else 1
    return 1 if n_fail else 0


def cmd_limits(args):
    import numpy as np

    from .limits.library import list_cable_types, load_cable_type
    if args.action == "list":
        for ct in list_cable_types(args.limit_dir):
            print(f"{ct.id:36s} {ct.status:24s} {ct.title}")
        return 0
    ct = load_cable_type(args.id, args.limit_dir)
    print(f"{ct.id}: {ct.title}\n  standard: {ct.standard}  clause: {ct.clause}\n  status: {ct.status}\n  provenance: {ct.provenance}")
    f = np.array([float(x) for x in (args.at or "1,10,100,600").split(",")]) * 1e6
    for lim in ct.limits:
        try:
            v, m = lim.evaluate(f, args.length)
        except Exception as e:
            print(f"  {lim.quantity}: {e}")
            continue
        vals = v if lim.kind != "range" else [f"{a:.2f}..{b:.2f}" for a, b in zip(*v)]
        print(f"  {lim.quantity:18s} {lim.kind:5s} {lim.unit:5s} " +
              "  ".join(f"{fi / 1e6:g}MHz:{(f'{x:.2f}' if isinstance(x, float) else x) if mm else '-'}"
                        for fi, x, mm in zip(f, vals, m)))
    return 0


def cmd_fixture(args):
    from .deembed import bisect_2x_thru, bisect_2x_thru_pair
    from .io import read_touchstone, write_touchstone
    thru = read_touchstone(args.thru)
    half = bisect_2x_thru_pair(thru, method=args.method) if thru.nports == 4 else bisect_2x_thru(thru, method=args.method)
    write_touchstone(half, args.out, comments=half.comments)
    print(f"wrote {args.out}: {half.comments[0]}")
    return 0


def cmd_db(args):
    from .db import ResultsDB
    with ResultsDB(args.db) as db:
        if args.action == "list":
            for r in db.runs(args.sample_id):
                hm = "" if r["headline_margin"] is None else f"{r['headline_margin']:+.2f} {r['headline_unit']}"
                print(f"{r['id']:4d} {r['timestamp']} {r['sample_id']:16s} {r['cable_type']:28s} {r['verdict']:4s} "
                      f"{r['headline_quantity'] or '':16s} {hm}")
        elif args.action == "show":
            for r in db.results_of(int(args.run)):
                print(dict(r))
        elif args.action == "export":
            print(db.export_csv(args.out or "results.csv"))
    return 0


def cmd_synth(args):
    from .reference import generate_reference_set
    paths = generate_reference_set(Path(args.out), n_points=args.points)
    for p in paths:
        print(p)
    return 0


def cmd_demo(args):
    """Generate the synthetic reference set, then evaluate every sample through the full
    pipeline: fixture removed, quantities computed, limits applied, reports and database written."""
    import json as _json

    from .reference import generate_reference_set
    out = Path(args.out)
    data = out / "data"
    print(f"==> synthesising the reference set into {data}")
    generate_reference_set(data, n_points=args.points)
    manifest = _json.loads((data / "manifest.json").read_text())
    fixture = {"method": "files", "left": str(data / manifest["fixture"]), "right": str(data / manifest["fixture"])}
    job = {"out": str(out / "reports"), "db": str(out / "results.sqlite"), "samples": []}
    for name, m in manifest["samples"].items():
        # the synthetic fixture is a 4-port launch, so it can only be removed from single-pair
        # measurements; the two-pair sample is evaluated with the fixture still in place
        n_ports = len(m["port_map"].split(","))
        fx = fixture if n_ports == 4 else {"method": "none"}
        if n_ports != 4:
            print(f"    {name}: {n_ports}-port measurement, evaluated with the fixture in place")
        job["samples"].append({
            "sample": {"sample_id": name, "cable_type": m["cable_type"], "length_m": m["length_m"],
                       "operator": "demo", "notes": "synthetic reference sample"},
            "files": [{"path": str(data / m["file"]), "port_map": m["port_map"], "fixture": fx}],
        })
    job_path = out / "job.json"
    job_path.write_text(_json.dumps(job, indent=1))
    print(f"==> evaluating {len(job['samples'])} samples")
    args.job, args.out, args.db, args.pdf = str(job_path), job["out"], job["db"], False
    rc = cmd_batch(args)
    print(f"\nreports in {job['out']}, database at {job['db']}")
    return rc


def cmd_gui(args):
    from .gui import main as gui_main
    gui_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cablecheck", description="Cable measurement-to-report pipeline")
    p.add_argument("--version", action="version", version=f"cablecheck {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evaluate", help="evaluate one sample from one or more Touchstone files")
    e.add_argument("files", nargs="+")
    e.add_argument("--cable-type", required=True, help="limit set id or TOML path")
    e.add_argument("--sample-id", required=True)
    e.add_argument("--length", type=float, help="sample length in m")
    e.add_argument("--port-map", action="append", help="per file, e.g. 'A+near,A-near,A+far,A-far' (repeatable)")
    e.add_argument("--fixture-left"), e.add_argument("--fixture-right")
    e.add_argument("--thru", help="2x-thru Touchstone for bisection de-embedding")
    e.add_argument("--port-extension-ps", type=float)
    e.add_argument("--nvp", type=float, help="nominal velocity of propagation for the distance axis")
    e.add_argument("--profile-engine", default="plain", choices=["plain", "zprofile"],
                   help="impedance profile: plain TDR transform, or the loss-aware reconstruction of the zprofile package")
    for k in ("lot", "part-number", "operator", "instrument", "instrument-serial", "calibration-date", "site", "notes"):
        e.add_argument(f"--{k}")
    e.add_argument("--temperature", type=float, help="degC")
    e.add_argument("--limit-dir", action="append", help="extra folder with limit TOML files")
    e.add_argument("--out", default="reports")
    e.add_argument("--pdf", action="store_true"), e.add_argument("--json", action="store_true")
    e.add_argument("--no-html", action="store_true")
    e.add_argument("--db", help="SQLite results database to append to")
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("batch", help="evaluate many samples from a JSON job file")
    b.add_argument("job"), b.add_argument("--out"), b.add_argument("--db"), b.add_argument("--pdf", action="store_true")
    b.set_defaults(func=cmd_batch)

    l = sub.add_parser("limits", help="list or show limit sets")
    l.add_argument("action", choices=["list", "show"])
    l.add_argument("id", nargs="?")
    l.add_argument("--at", help="frequencies in MHz, comma separated")
    l.add_argument("--length", type=float)
    l.add_argument("--limit-dir", action="append")
    l.set_defaults(func=cmd_limits)

    f = sub.add_parser("fixture", help="split a 2x-thru into a half fixture")
    f.add_argument("thru"), f.add_argument("-o", "--out", required=True)
    f.add_argument("--method", default="auto", choices=["auto", "gated", "symmetric"])
    f.set_defaults(func=cmd_fixture)

    d = sub.add_parser("db", help="query the results database")
    d.add_argument("action", choices=["list", "show", "export"])
    d.add_argument("--db", required=True), d.add_argument("--sample-id"), d.add_argument("--run"), d.add_argument("--out")
    d.set_defaults(func=cmd_db)

    s = sub.add_parser("synth", help="generate the synthetic reference measurement set")
    s.add_argument("--out", default="reference"), s.add_argument("--points", type=int, default=1200)
    s.set_defaults(func=cmd_synth)

    dm = sub.add_parser("demo", help="generate the reference set and evaluate it end to end")
    dm.add_argument("out", nargs="?", default="demo_out")
    dm.add_argument("--points", type=int, default=1200)
    dm.set_defaults(func=cmd_demo)

    g = sub.add_parser("gui", help="open the lab interface")
    g.set_defaults(func=cmd_gui)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
