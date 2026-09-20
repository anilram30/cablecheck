import numpy as np

from cablecheck.deembed import (
    bisect_2x_thru,
    bisect_2x_thru_pair,
    deembed,
    embed,
    port_extension,
)
from cablecheck.mixedmode import PortMap, to_mixed_mode
from cablecheck.network import Network, cascade, reverse_ports
from cablecheck.synth import FixtureSpec, PairSpec, make_fixture, make_pair


def test_embed_then_deembed_is_identity(f_grid):
    dut = make_pair(PairSpec(length_m=5.0, n_segments=10, ripple_amp=0.02), f_grid)
    fx = make_fixture(FixtureSpec(), f_grid)
    meas = embed(dut, fx, fx)
    assert np.max(np.abs(meas.s - dut.s)) > 1e-3  # the fixture does something
    back = deembed(meas, fx, fx)
    assert np.allclose(back.s, dut.s, atol=1e-10)
    # one-sided
    assert np.allclose(deembed(embed(dut, fx, None), fx, None).s, dut.s, atol=1e-10)


def test_port_extension_removes_pure_delay(f_grid):
    dut = make_pair(PairSpec(length_m=2.0, n_segments=4), f_grid)
    tau = 120e-12
    ph = np.exp(-1j * 2 * np.pi * f_grid * tau)
    s = dut.s * ph[:, None, None] * ph[:, None, None]
    ext = port_extension(Network(f_grid, s, 50.0), tau)
    assert np.allclose(ext.s, dut.s, atol=1e-12)


def _ideal_thru4(f):
    s = np.tile(np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], complex)[None], (f.size, 1, 1))
    return Network(f, s, 50.0)


def test_symmetric_bisection_is_exact_for_symmetric_half(f_grid):
    # a fixture without the lumped connector is symmetric in itself
    fx = make_fixture(FixtureSpec(conn_series_l=0.0, conn_shunt_c=0.0), f_grid)
    thru = embed(_ideal_thru4(f_grid), fx, fx)
    half = bisect_2x_thru_pair(thru, method="symmetric")
    assert np.allclose(half.s, fx.s, atol=1e-6)


def test_auto_bisection_on_short_fixture_is_close(f_grid):
    fx = make_fixture(FixtureSpec(), f_grid)
    thru = embed(_ideal_thru4(f_grid), fx, fx)
    half = bisect_2x_thru_pair(thru)
    assert "symmetric" in half.comments[0]
    a, b = to_mixed_mode(fx, PortMap.single_pair()), to_mixed_mode(half, PortMap.single_pair())
    err = np.max(np.abs(a.s - b.s))
    asym = np.max(np.abs(a.param(("d", "A", "near"), ("d", "A", "near")) - a.param(("d", "A", "far"), ("d", "A", "far"))))
    assert err < 1.5 * asym + 1e-6  # error bounded by the half's own asymmetry


def test_gated_bisection_when_resolved():
    f = np.linspace(1e6, 20e9, 1500)
    fx = make_fixture(FixtureSpec(), f)
    thru = embed(_ideal_thru4(f), fx, fx)
    half = bisect_2x_thru_pair(thru)
    assert "gated" in half.comments[0]
    a, b = to_mixed_mode(fx, PortMap.single_pair()), to_mixed_mode(half, PortMap.single_pair())
    m = f < 4e9  # where the fixture is still a small reflection
    for lab in [(("d", "A", "near"), ("d", "A", "near")), (("d", "A", "far"), ("d", "A", "near"))]:
        assert np.max(np.abs(a.param(*lab) - b.param(*lab))[m]) < 0.04


def test_two_port_bisection_roundtrip(f_grid):
    fx = make_fixture(FixtureSpec(conn_series_l=0.0, conn_shunt_c=0.0), f_grid)
    fx2 = fx.subnetwork([0, 2])  # one wire
    thru = Network(f_grid, cascade(fx2.s, reverse_ports(fx2.s)), 50.0)
    half = bisect_2x_thru(thru, method="symmetric")
    assert np.allclose(half.s, fx2.s, atol=1e-6)
