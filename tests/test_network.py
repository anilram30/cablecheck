import numpy as np

from cablecheck.network import (
    Network,
    abcd_to_s,
    cascade,
    renormalize,
    reverse_ports,
    s_to_t,
    s_to_z,
    t_to_s,
    z_to_s,
)
from cablecheck.synth import C0, chain_line


def lossless_line_s(f, z_line, length, v, z0=50.0):
    """Analytic S of a lossless line of impedance z_line in a z0 system."""
    g = 1j * 2 * np.pi * f / v
    den = 2 * z_line * z0 * np.cosh(g * length) + (z_line ** 2 + z0 ** 2) * np.sinh(g * length)
    s11 = (z_line ** 2 - z0 ** 2) * np.sinh(g * length) / den
    s21 = 2 * z_line * z0 / den
    s = np.zeros((f.size, 2, 2), complex)
    s[:, 0, 0] = s[:, 1, 1] = s11
    s[:, 0, 1] = s[:, 1, 0] = s21
    return s


def test_abcd_to_s_matches_analytic_line(f_grid):
    f = f_grid
    z, v, l = 75.0, 0.7 * C0, 2.0
    w = 2 * np.pi * f
    Z = (1j * w * z / v)[:, None, None] * np.eye(1)[None]
    Y = (1j * w / (z * v))[:, None, None] * np.eye(1)[None]
    s = abcd_to_s(chain_line(Z, Y, l), np.array([50.0, 50.0]))
    assert np.allclose(s, lossless_line_s(f, z, l, v), atol=1e-10)


def test_z_roundtrip_and_renormalisation(f_grid):
    s = lossless_line_s(f_grid, 75.0, 1.0, 0.7 * C0)
    z0 = np.array([50.0, 50.0])
    assert np.allclose(z_to_s(s_to_z(s, z0), z0), s, atol=1e-12)
    # renormalising to the line impedance makes it a pure delay
    s75 = renormalize(s, z0, np.array([75.0, 75.0]))
    assert np.allclose(np.abs(s75[:, 0, 0]), 0, atol=1e-10)
    assert np.allclose(np.abs(s75[:, 1, 0]), 1, atol=1e-10)
    # and back
    assert np.allclose(renormalize(s75, [75.0, 75.0], z0), s, atol=1e-10)


def test_t_parameters_cascade_lines(f_grid):
    v = 0.7 * C0
    a = lossless_line_s(f_grid, 75.0, 1.0, v)
    b = lossless_line_s(f_grid, 75.0, 2.5, v)
    ab = lossless_line_s(f_grid, 75.0, 3.5, v)
    assert np.allclose(cascade(a, b), ab, atol=1e-9)
    assert np.allclose(t_to_s(s_to_t(a)), a, atol=1e-12)


def test_reverse_ports_swaps_groups():
    s = np.arange(16, dtype=complex).reshape(1, 4, 4)
    r = reverse_ports(s)
    assert np.allclose(r[0, :2, :2], s[0, 2:, 2:])
    assert np.allclose(r[0, :2, 2:], s[0, 2:, :2])
    assert np.allclose(reverse_ports(r), s)


def test_network_helpers(f_grid):
    s = lossless_line_s(f_grid, 75.0, 1.0, 0.7 * C0)
    n = Network(f_grid, s, 50.0, name="line")
    assert n.nports == 2 and n.nfreq == f_grid.size
    assert np.all(n.passivity_violation() < 1e-9)
    assert np.all(n.reciprocity_error() < 1e-12)
    c = n.crop(10e6, 100e6)
    assert c.f.min() >= 10e6 and c.f.max() <= 100e6
    i = n.interpolate(np.linspace(2e6, 500e6, 50))
    assert i.nfreq == 50
    sub = n.subnetwork([1])
    assert sub.nports == 1 and np.allclose(sub.s[:, 0, 0], s[:, 1, 1])
