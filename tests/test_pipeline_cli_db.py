import json
import sqlite3
from pathlib import Path

from cablecheck.cli import main
from cablecheck.db import ResultsDB
from cablecheck.pipeline import (
    FixtureSpec,
    MeasurementFile,
    SampleInfo,
    run_sample,
    write_json,
)
from cablecheck.report.html import write_html
from cablecheck.report.pdf import write_pdf

REF = Path(__file__).parent / "reference"


def _run(tmp_path):
    s = SampleInfo("good_15m", "1000base-t1-link-segment", 15.0, lot="L1", operator="op", instrument="VNA",
                   calibration_date="2026-09-01", temperature_c=23.0)
    return run_sample(s, [MeasurementFile(str(REF / "good_15m.s4p"), fixture=FixtureSpec("files", left=str(REF / "fixture_launch.s4p")))])


def test_reports_and_db(tmp_path):
    res = _run(tmp_path)
    assert res.verdict == "PASS"
    h = write_html(res, tmp_path / "r.html")
    txt = h.read_text()
    assert "PASS" in txt and "<svg" in txt and "sha256" in txt
    p = write_pdf(res, tmp_path / "r.pdf")
    assert p.stat().st_size > 10_000
    j = write_json(res, tmp_path / "r.json")
    d = json.loads(j.read_text())
    assert d["verdict"] == "PASS" and d["sample"]["lot"] == "L1" and d["files"][0]["sha256"]
    assert any(t["quantity"] == "insertion_loss" for t in d["traces"])
    with ResultsDB(tmp_path / "db.sqlite") as db:
        rid = db.insert_run(res, str(h))
        rid2 = db.insert_run(res, str(h))
        assert rid2 == rid + 1
        runs = db.runs("good_15m")
        assert len(runs) == 2 and runs[0]["verdict"] == "PASS"
        rows = db.results_of(rid)
        assert any(r["quantity"] == "return_loss" for r in rows)
        assert db.run_json(rid)["sample"]["operator"] == "op"
        csv = db.export_csv(tmp_path / "x.csv")
        assert csv.read_text().count("\n") > 10
    con = sqlite3.connect(tmp_path / "db.sqlite")
    assert con.execute("select count(*) from samples").fetchone()[0] == 1
    assert con.execute("select count(*) from files").fetchone()[0] == 2


def test_cli_end_to_end(tmp_path, capsys):
    rc = main(["evaluate", str(REF / "lossy_15m.s4p"), "--cable-type", "1000base-t1-link-segment", "--sample-id", "lossy",
               "--length", "15", "--fixture-left", str(REF / "fixture_launch.s4p"), "--out", str(tmp_path),
               "--json", "--db", str(tmp_path / "r.sqlite"), "--no-html"])
    out = capsys.readouterr().out
    assert rc == 1 and "FAIL" in out and "insertion_loss" in out
    assert list(tmp_path.glob("lossy_*.json"))
    rc = main(["db", "list", "--db", str(tmp_path / "r.sqlite")])
    assert rc == 0 and "lossy" in capsys.readouterr().out
    rc = main(["limits", "show", "1000base-t1-link-segment", "--at", "10,100"])
    assert rc == 0 and "insertion_loss" in capsys.readouterr().out
    rc = main(["fixture", str(REF / "fixture_2xthru.s4p"), "-o", str(tmp_path / "half.s4p")])
    assert rc == 0 and (tmp_path / "half.s4p").exists()


def test_batch_job(tmp_path, capsys):
    job = {"out": str(tmp_path / "rep"), "db": str(tmp_path / "b.sqlite"),
           "samples": [{"sample": {"sample_id": "good", "cable_type": "1000base-t1-link-segment", "length_m": 15.0},
                        "files": [{"path": str(REF / "good_15m.s4p"),
                                   "fixture": {"method": "files", "left": str(REF / "fixture_launch.s4p")}}]},
                       {"sample": {"sample_id": "two", "cable_type": "1000base-t1-link-segment", "length_m": 5.0},
                        "files": [{"path": str(REF / "twopair_5m_truth.s8p"),
                                   "port_map": "A+near,A-near,A+far,A-far,B+near,B-near,B+far,B-far"}]}]}
    (tmp_path / "job.json").write_text(json.dumps(job))
    # exit code is the batch verdict: non-zero when any sample fails its limits (one does here)
    assert main(["batch", str(tmp_path / "job.json")]) == 1
    out = capsys.readouterr().out
    assert "sample good" in out and "sample two" in out and "next" in out
    assert len(list((tmp_path / "rep").glob("*.html"))) == 2


def test_missing_quantity_warning(tmp_path):
    s = SampleInfo("x", "1000base-t1-link-segment", 15.0)
    res = run_sample(s, [MeasurementFile(str(REF / "good_15m_truth.s4p"))])
    assert any("next" in w for w in res.evaluation.warnings)
