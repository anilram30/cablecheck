"""
Results database (SQLite, standard library only).

Schema
------
samples   one row per physical sample (sample_id + cable type + build data)
runs      one evaluation of one sample: who/what/when, verdict, headline, report path,
          software version, JSON blob with the full result (traces included)
files     the raw files of a run with sha256 - the audit trail
results   one row per (quantity, pair) of a run with its worst margin

Everything an auditor needs to reproduce a verdict six months later is in
these four tables: the input hashes, the limit file id and status, the code
version and the numbers themselves.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .pipeline import RunResult

__all__ = ["ResultsDB"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY,
    sample_id TEXT NOT NULL,
    cable_type TEXT NOT NULL,
    part_number TEXT, lot TEXT, length_m REAL, notes TEXT,
    UNIQUE(sample_id, cable_type)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    sample_ref INTEGER NOT NULL REFERENCES samples(id),
    timestamp TEXT NOT NULL,
    operator TEXT, instrument TEXT, instrument_serial TEXT, calibration_date TEXT,
    temperature_c REAL, humidity_pct REAL, site TEXT,
    limit_file TEXT, limit_status TEXT,
    software_version TEXT,
    verdict TEXT NOT NULL,
    headline_quantity TEXT, headline_pair TEXT, headline_margin REAL, headline_x REAL, headline_unit TEXT,
    report_path TEXT,
    json TEXT
);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    run_ref INTEGER NOT NULL REFERENCES runs(id),
    name TEXT, path TEXT, sha256 TEXT, nports INTEGER, fmin_hz REAL, fmax_hz REAL, npoints INTEGER,
    port_map TEXT, fixture TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    run_ref INTEGER NOT NULL REFERENCES runs(id),
    quantity TEXT NOT NULL, pair TEXT NOT NULL, unit TEXT, kind TEXT,
    passed INTEGER, worst_margin REAL, x_worst REAL, value_at_worst REAL, limit_at_worst REAL,
    n_points INTEGER, n_fail INTEGER
);
CREATE INDEX IF NOT EXISTS ix_results_run ON results(run_ref);
CREATE INDEX IF NOT EXISTS ix_runs_sample ON runs(sample_ref);
"""


class ResultsDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(_SCHEMA)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------- writes
    def insert_run(self, result: RunResult, report_path: str | None = None, store_traces: bool = True) -> int:
        s = result.sample
        cur = self.con.cursor()
        cur.execute("INSERT OR IGNORE INTO samples(sample_id, cable_type, part_number, lot, length_m, notes) VALUES (?,?,?,?,?,?)",
                    (s.sample_id, result.cable_type.id, s.part_number, s.lot, s.length_m, s.notes))
        cur.execute("SELECT id FROM samples WHERE sample_id=? AND cable_type=?", (s.sample_id, result.cable_type.id))
        sample_ref = cur.fetchone()["id"]
        h = result.evaluation.headline
        cur.execute(
            "INSERT INTO runs(sample_ref, timestamp, operator, instrument, instrument_serial, calibration_date, temperature_c, "
            "humidity_pct, site, limit_file, limit_status, software_version, verdict, headline_quantity, headline_pair, "
            "headline_margin, headline_x, headline_unit, report_path, json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sample_ref, result.timestamp, s.operator, s.instrument, s.instrument_serial, s.calibration_date,
             s.temperature_c, s.humidity_pct, s.site, result.cable_type.path, result.cable_type.status,
             result.software_version, result.verdict,
             h.quantity if h else None, h.pair if h else None, h.worst_margin if h else None,
             h.x_worst if h else None, h.unit if h else None, report_path,
             json.dumps(result.to_json_dict(include_traces=store_traces))))
        run_id = cur.lastrowid
        for fr in result.files:
            cur.execute("INSERT INTO files(run_ref, name, path, sha256, nports, fmin_hz, fmax_hz, npoints, port_map, fixture) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (run_id, fr["name"], fr["path"], fr["sha256"], fr["nports"], fr["fmin_hz"], fr["fmax_hz"],
                         fr["npoints"], fr["port_map"], fr["fixture"]))
        for r in result.evaluation.results:
            cur.execute("INSERT INTO results(run_ref, quantity, pair, unit, kind, passed, worst_margin, x_worst, value_at_worst, "
                        "limit_at_worst, n_points, n_fail) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (run_id, r.quantity, r.pair, r.unit, r.kind, None if r.passed is None else int(r.passed),
                         r.worst_margin, r.x_worst, r.value_at_worst, r.limit_at_worst, r.n_points, r.n_fail))
        self.con.commit()
        return run_id

    # -------------------------------------------------------------- reads
    def runs(self, sample_id: str | None = None, limit: int = 100) -> list[sqlite3.Row]:
        q = ("SELECT runs.id, samples.sample_id, samples.cable_type, runs.timestamp, runs.operator, runs.verdict, "
             "runs.headline_quantity, runs.headline_pair, runs.headline_margin, runs.headline_x, runs.headline_unit, "
             "runs.temperature_c, runs.report_path FROM runs JOIN samples ON runs.sample_ref = samples.id")
        args: tuple = ()
        if sample_id:
            q += " WHERE samples.sample_id = ?"
            args = (sample_id,)
        q += " ORDER BY runs.timestamp DESC LIMIT ?"
        return self.con.execute(q, args + (limit,)).fetchall()

    def results_of(self, run_id: int) -> list[sqlite3.Row]:
        return self.con.execute("SELECT * FROM results WHERE run_ref=? ORDER BY worst_margin", (run_id,)).fetchall()

    def run_json(self, run_id: int) -> dict:
        row = self.con.execute("SELECT json FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row["json"])

    def export_csv(self, path: str | Path) -> Path:
        import csv
        path = Path(path)
        rows = self.con.execute(
            "SELECT samples.sample_id, samples.cable_type, samples.lot, samples.length_m, runs.timestamp, runs.operator, "
            "runs.instrument, runs.calibration_date, runs.temperature_c, runs.verdict, results.quantity, results.pair, "
            "results.kind, results.passed, results.worst_margin, results.x_worst, results.value_at_worst, results.limit_at_worst, "
            "results.unit, results.n_fail, results.n_points FROM results JOIN runs ON results.run_ref = runs.id "
            "JOIN samples ON runs.sample_ref = samples.id ORDER BY runs.timestamp").fetchall()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if rows:
                w.writerow(rows[0].keys())
            for r in rows:
                w.writerow(list(r))
        return path
