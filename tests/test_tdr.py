import numpy as np

from cablecheck.network import abcd_to_s, cascade
from cablecheck.synth import C0, chain_line
from cablecheck.tdr import gate, impedance_from_step, to_time_domain


def line2(f, z, length, v):
    w = 2 * np.pi * f
    Z = (1j * w * z / v)[:, None, None] * np.eye(1)[None]
    Y = (1j * w / (z * v))[:, None, None] * np.eye(1)[None]
    return abcd_to_s(chain_line(Z, Y, length), np.array([100.0, 100.0]))


def test_impedance_step_is_located_and_valued():
    f = np.linspace(1e6, 3e9, 1500)
    v = 0.68 * C0
    s = cascade(line2(f, 100.0, 2.0, v), line2(f, 115.0, 2.0, v))
    tr = to_time_domain(f, s[:, 0, 0], t_rise=100e-12)
    z = impedance_from_step(tr.step, 100.0)
    x = tr.distance(0.68)
    assert abs(np.mean(z[(x > 0.5) & (x < 1.5)]) - 100) < 0.5
    assert abs(np.mean(z[(x > 2.5) & (x < 3.5)]) - 115) < 0.8
    # transition near 2 m
    k = np.argmax(z > 107.5)
    assert abs(x[k] - 2.0) < 0.1


def test_gate_isolates_first_reflection():
    f = np.linspace(1e6, 3e9, 1500)
    v = 0.68 * C0
    first = line2(f, 110.0, 1.0, v)          # 100 -> 110 step at the input, back to 100 at 1 m
    s = cascade(first, line2(f, 100.0, 0.5, v), line2(f, 130.0, 1.0, v))
    t_end = 2 * 1.2 / v                       # round trip to just beyond the first section
    g = gate(f, s[:, 0, 0], -1e-9, t_end)
    assert np.max(np.abs(g - first[:, 0, 0])[f < 2e9]) < 0.03


def test_dc_extrapolation_is_real_and_bounded():
    f = np.linspace(50e6, 1e9, 200)
    s11 = 0.05 * np.exp(-1j * 2 * np.pi * f * 1e-9)
    tr = to_time_domain(f, s11)
    assert tr.f_uniform[0] == 0.0
    assert abs(tr.s_uniform[0].imag) < 1e-12 and abs(tr.s_uniform[0]) <= 1
