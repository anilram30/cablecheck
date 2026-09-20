import numpy as np

from cablecheck.mixedmode import PortMap, to_mixed_mode
from cablecheck.quantities import compute_quantities, fitted_impedance, phase_delay
from cablecheck.synth import C0, PairSpec, make_pair, make_two_pairs


def _get(traces, q, pair="A"):
    return next(t for t in traces if t.quantity == q and t.pair == pair)


def test_delay_and_loss_against_model(f_grid):
    spec = PairSpec(length_m=8.0, n_segments=8)
    n = make_pair(spec, f_grid)
    tr = compute_quantities(n, PortMap.single_pair(), length_m=8.0)
    tau = _get(tr, "phase_delay").y * 1e-9
    assert abs(np.median(tau) - 8.0 / (spec.nvp * C0)) < 0.05e-9
    nvp = _get(tr, "nvp").y[0]
    assert abs(nvp - spec.nvp) < 0.01
    il = _get(tr, "insertion_loss").y
    assert np.all(np.diff(il[f_grid > 20e6]) > -0.05)  # monotone-ish increasing
    assert il[0] < 0.5 and 3 < il[-1] < 8


def test_phase_delay_unwrap_for_long_cable():
    f = np.linspace(1e6, 600e6, 3000)
    n = make_pair(PairSpec(length_m=40.0, n_segments=10), f)
    mm = to_mixed_mode(n, PortMap.single_pair())
    tau = phase_delay(f, mm.param(("d", "A", "far"), ("d", "A", "near")))
    assert abs(np.median(tau) - 40.0 / (0.68 * C0)) < 0.1e-9


def test_fitted_impedance_recovers_nominal():
    f = np.linspace(1e6, 600e6, 801)
    for z in (95.0, 100.0, 108.0):
        n = make_pair(PairSpec(length_m=12.0, z_diff=z, n_segments=12), f)
        mm = to_mixed_mode(n, PortMap.single_pair())
        a, b, zm = fitted_impedance(f, mm.param(("d", "A", "near"), ("d", "A", "near")), 100.0)
        assert abs(zm - z) < 0.3


def test_two_pair_crosstalk_quantities(f_grid):
    n = make_two_pairs(PairSpec(length_m=3.0, n_segments=6), PairSpec(length_m=3.0, n_segments=6), f_grid,
                       coupling_l=0.02, coupling_c=0.015, imbalance=0.15)
    tr = compute_quantities(n, PortMap.two_pairs(), length_m=3.0)
    names = {(t.quantity, t.pair) for t in tr}
    assert ("next", "A->B") in names and ("fext", "A->B") in names and ("delay_skew", "all") in names
    nx = _get(tr, "next", "A->B").y
    assert np.all(nx > 20) and nx[-1] < nx[0]          # crosstalk grows with frequency
    assert np.all(_get(tr, "delay_skew", "all").y < 0.05)
