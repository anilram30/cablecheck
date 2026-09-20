"""
Golden-file regression: every reference sample is evaluated through the full
pipeline (explicit fixture de-embedding) and every reported number is compared
with ``tests/reference/expected.json``.

Refresh deliberately with  CABLECHECK_UPDATE_EXPECTED=1 pytest tests/test_regression.py
and review the diff before committing.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from cablecheck.pipeline import FixtureSpec, MeasurementFile, SampleInfo, run_sample

REF = Path(__file__).parent / "reference"
EXPECTED = REF / "expected.json"
TOL = {"dB": 1e-3, "ns": 1e-3, "ohm": 1e-3, "ns/m": 1e-4, "": 1e-4}


def _manifest():
    return json.loads((REF / "manifest.json").read_text())


def _run(name, entry, fixture="files"):
    fx = FixtureSpec("files", left=str(REF / "fixture_launch.s4p")) if fixture == "files" else \
        FixtureSpec("2xthru", thru=str(REF / "fixture_2xthru.s4p"))
    if entry["file"].endswith("s8p"):
        fx = FixtureSpec()  # 8-port fixture handled by the truth comparison instead
    sample = SampleInfo(sample_id=name, cable_type=entry["cable_type"], length_m=entry["length_m"], operator="regression")
    return run_sample(sample, [MeasurementFile(str(REF / entry["file"]), entry["port_map"], fx)])


def _snapshot(res):
    return {"verdict": res.verdict,
            "results": {f"{r.quantity}[{r.pair}]": {"passed": r.passed, "worst_margin": r.worst_margin,
                                                   "x_worst": r.x_worst, "value_at_worst": r.value_at_worst,
                                                   "limit_at_worst": r.limit_at_worst, "n_fail": r.n_fail,
                                                   "n_points": r.n_points, "unit": r.unit,
                                                   "scalar_value": float(r.value[0]) if r.is_scalar else None}
                        for r in res.evaluation.results}}


@pytest.fixture(scope="module")
def expected():
    if EXPECTED.exists():
        return json.loads(EXPECTED.read_text())
    return {}


@pytest.mark.parametrize("name", list(_manifest()["samples"]))
def test_reference_sample(name, expected):
    entry = _manifest()["samples"][name]
    res = _run(name, entry)
    snap = _snapshot(res)
    if os.environ.get("CABLECHECK_UPDATE_EXPECTED"):
        expected[name] = snap
        EXPECTED.write_text(json.dumps(expected, indent=1, sort_keys=True))
        pytest.skip("expected results updated")
    assert name in expected, "no frozen expectation - run with CABLECHECK_UPDATE_EXPECTED=1"
    exp = expected[name]
    assert snap["verdict"] == exp["verdict"]
    assert set(snap["results"]) == set(exp["results"])
    for key, e in exp["results"].items():
        got = snap["results"][key]
        tol = TOL.get(e["unit"], 1e-3)
        assert got["passed"] == e["passed"], key
        assert got["n_fail"] == e["n_fail"] and got["n_points"] == e["n_points"], key
        for fld in ("worst_margin", "value_at_worst", "limit_at_worst", "scalar_value"):
            if e[fld] is None:
                assert got[fld] is None, (key, fld)
            else:
                assert abs(got[fld] - e[fld]) <= tol, (key, fld, got[fld], e[fld])
        if e["x_worst"] is not None:
            assert abs(got["x_worst"] - e["x_worst"]) <= 1e-6 * max(1.0, abs(e["x_worst"])), key


def test_expected_verdicts_by_design(expected):
    """The reference set was designed with these outcomes; guards the data set itself."""
    if not expected:
        pytest.skip("no frozen expectation yet")
    by = {k: v["results"] for k, v in expected.items()}
    assert expected["good_15m"]["verdict"] == "PASS"
    assert not by["lossy_15m"]["insertion_loss[A]"]["passed"]
    assert not by["ripple_15m"]["return_loss[A]"]["passed"]
    assert not by["unbalanced_10m"]["lcl[A]"]["passed"] and not by["unbalanced_10m"]["lctl[A]"]["passed"]
    assert "next[A->B]" in by["twopair_5m"] and "fext[A->B]" in by["twopair_5m"]


@pytest.mark.parametrize("name", ["good_15m", "ripple_15m"])
def test_deembedding_recovers_truth(name):
    """De-embedded measurement must match the bare-DUT truth file within the noise."""
    from cablecheck.deembed import deembed
    from cablecheck.io import read_touchstone
    from cablecheck.mixedmode import PortMap, to_mixed_mode
    from cablecheck.network import db
    entry = _manifest()["samples"][name]
    meas = read_touchstone(REF / entry["file"])
    truth = read_touchstone(REF / entry["truth"])
    fx = read_touchstone(REF / "fixture_launch.s4p")
    d = deembed(meas, fx, fx)
    a, b = to_mixed_mode(d, PortMap.single_pair()), to_mixed_mode(truth, PortMap.single_pair())
    il_a = -db(a.param(("d", "A", "far"), ("d", "A", "near")))
    il_b = -db(b.param(("d", "A", "far"), ("d", "A", "near")))
    assert np.max(np.abs(il_a - il_b)) < 0.02
    rl_a = -db(a.param(("d", "A", "near"), ("d", "A", "near")))
    rl_b = -db(b.param(("d", "A", "near"), ("d", "A", "near")))
    m = rl_b < 40  # where the reflection is above the noise floor
    assert np.max(np.abs(rl_a - rl_b)[m]) < 0.5


def test_2xthru_path_agrees_with_explicit_fixture():
    entry = _manifest()["samples"]["good_15m"]
    a = _run("good_15m", entry, fixture="files")
    b = _run("good_15m", entry, fixture="2xthru")
    ra = {r.label: r for r in a.evaluation.results}
    rb = {r.label: r for r in b.evaluation.results}
    assert abs(ra["insertion_loss[A]"].worst_margin - rb["insertion_loss[A]"].worst_margin) < 0.05
    assert a.verdict == b.verdict == "PASS"
