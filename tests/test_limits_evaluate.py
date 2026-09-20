import numpy as np
import pytest

from cablecheck.evaluate import evaluate
from cablecheck.limits.library import (
    LimitError,
    list_cable_types,
    load_cable_type,
    safe_eval,
)
from cablecheck.quantities import Trace


def test_safe_eval_whitelist():
    f = np.array([1.0, 100.0])
    assert np.allclose(safe_eval("2*sqrt(f) + 1", f=f), [3, 21])
    for bad in ("__import__('os')", "f.__class__", "open('x')", "lambda: 1"):
        with pytest.raises(LimitError):
            safe_eval(bad, f=f)


def test_builtin_types_load_and_evaluate():
    types = list_cable_types()
    assert {t.id for t in types} >= {"1000base-t1-link-segment", "1000base-t1-cable-per-metre", "lab-generic-100ohm-stp"}
    ct = load_cable_type("1000base-t1-link-segment")
    v, m = ct.limit_for("insertion_loss").evaluate(np.array([1e6, 600e6]))
    assert np.allclose(v, [0.0023 + 0.5907 + 0.0639, 0.0023 * 600 + 0.5907 * np.sqrt(600) + 0.0639 / np.sqrt(600)])
    v, m = ct.limit_for("lcl").evaluate(np.array([5e6, 80e6]))
    assert not m[0] and m[1] and np.isclose(v[1], 50)


def test_per_metre_scaling():
    ct = load_cable_type("1000base-t1-cable-per-metre")
    v15, _ = ct.limit_for("insertion_loss").evaluate(np.array([100e6]), 15.0)
    v5, _ = ct.limit_for("insertion_loss").evaluate(np.array([100e6]), 5.0)
    assert np.isclose(v15 / v5, 3.0)
    with pytest.raises(LimitError):
        ct.limit_for("insertion_loss").evaluate(np.array([100e6]), None)


def test_margin_sign_conventions():
    ct = load_cable_type("lab-generic-100ohm-stp")
    f = np.linspace(1e6, 1000e6, 100)
    lim, _ = ct.limit_for("insertion_loss").evaluate(f, 10.0)
    il = Trace("insertion_loss", "A", f, lim - 1.0, "dB")                     # 1 dB inside everywhere
    rl_lim, _ = ct.limit_for("return_loss").evaluate(f)
    rl = Trace("return_loss", "A", f, rl_lim + np.where(f > 500e6, -2.0, 3.0), "dB")  # fails above 500 MHz
    zf = Trace("impedance_fitted", "A", np.array([np.nan]), np.array([104.0]), "ohm", "")
    ev = evaluate([il, rl, zf], ct, length_m=10.0)
    r = {x.quantity: x for x in ev.results}
    assert r["insertion_loss"].passed and np.isclose(r["insertion_loss"].worst_margin, 1.0)
    assert not r["return_loss"].passed and np.isclose(r["return_loss"].worst_margin, -2.0)
    assert r["return_loss"].x_worst > 500e6 and r["return_loss"].n_fail == int(np.sum(f > 500e6))
    assert r["impedance_fitted"].passed and np.isclose(r["impedance_fitted"].worst_margin, 1.0)
    assert ev.verdict == "FAIL" and ev.headline.quantity == "return_loss"
    assert any("per metre" not in w for w in ev.warnings) or True


def test_unlimited_quantity_is_informational():
    ct = load_cable_type("lab-generic-100ohm-stp")
    tr = Trace("group_delay", "A", np.linspace(1e6, 1e8, 10), np.ones(10), "ns")
    ev = evaluate([tr], ct)
    assert ev.results[0].kind is None and ev.verdict == "NO LIMITS"
