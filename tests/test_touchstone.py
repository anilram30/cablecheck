import numpy as np
import pytest

from cablecheck.io import (
    TouchstoneError,
    parse_touchstone,
    read_touchstone,
    write_touchstone,
)
from cablecheck.network import Network


def _rand_net(n, nf=7, seed=0):
    rng = np.random.default_rng(seed)
    f = np.linspace(1e6, 1e9, nf)
    s = 0.3 * (rng.standard_normal((nf, n, n)) + 1j * rng.standard_normal((nf, n, n)))
    return Network(f, s, 50.0)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("fmt", ["RI", "MA", "DB"])
def test_roundtrip(tmp_path, n, fmt):
    net = _rand_net(n)
    p = write_touchstone(net, tmp_path / f"x.s{n}p", fmt=fmt, unit="MHz")
    back = read_touchstone(p)
    assert back.nports == n
    assert np.allclose(back.f, net.f)
    assert np.allclose(back.s, net.s, atol=1e-9)
    assert np.allclose(back.z0, 50.0)


def test_two_port_ordering_quirk():
    # v1 2-port files list S11 S21 S12 S22
    txt = "# MHz S RI R 50\n1 0.1 0 0.9 0 0.8 0 0.2 0\n2 0.1 0 0.9 0 0.8 0 0.2 0\n"
    n = parse_touchstone(txt, nports_hint=2)
    assert np.isclose(n.s[0, 1, 0], 0.9) and np.isclose(n.s[0, 0, 1], 0.8)
    assert np.isclose(n.f[0], 1e6)


def test_v2_lower_matrix():
    txt = ("[Version] 2.0\n# GHz S MA R 50\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n"
           "[Number of Frequencies] 1\n[Matrix Format] Lower\n[Reference] 50 75\n[Network Data]\n"
           "1.0 0.5 0 0.9 -90 0.4 0\n[End]\n")
    n = parse_touchstone(txt)
    assert n.nports == 2 and np.isclose(n.f[0], 1e9)
    assert np.allclose(n.z0, [50, 75])
    assert np.isclose(n.s[0, 1, 0], -0.9j) and np.isclose(n.s[0, 0, 1], -0.9j)
    assert np.isclose(n.s[0, 1, 1], 0.4)


def test_rejects_non_s_files():
    with pytest.raises(TouchstoneError):
        parse_touchstone("# MHz Z RI R 50\n1 1 0 0 0 0 0 1 0\n", nports_hint=2)


def test_comments_and_noise_block_ignored():
    txt = ("! test\n# MHz S RI R 50\n1 0.1 0 0.9 0 0.9 0 0.1 0 ! inline\n2 0.1 0 0.9 0 0.9 0 0.1 0\n"
           "! noise parameters follow\n1 2.0 0.5 0 0.3\n")
    n = parse_touchstone(txt, nports_hint=2)
    assert n.nfreq == 2 and "test" in n.comments[0]
