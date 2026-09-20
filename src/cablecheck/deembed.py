"""
Fixture removal.

Three methods, in decreasing order of rigour:

1. **Explicit fixture de-embedding** (:func:`deembed`).  With the measured
   2n-port  T_meas = T_L T_DUT T_R'  (block wave-chain matrices, R' = the
   right fixture viewed from the DUT side) the cable is recovered exactly as
   T_DUT = T_L^{-1} T_meas T_R'^{-1}.  Fixture files are ordinary Touchstone
   networks whose ports 1..n are the VNA side and n+1..2n the DUT side.

2. **2x-thru bisection** (:func:`bisect_2x_thru`).  When only a
   "fixture-fixture" (2x-thru) measurement is available, each half is
   recovered under the mirror-symmetry assumption B = reverse(A).  Two
   estimators are provided and chosen automatically from the product of
   the 2x-thru delay T and the sweep bandwidth B:

   * ``gated`` (T B >= 4, i.e. the two halves are resolved in time):
         S11_2x = S11a + S21a^2 S22a / (1 - S22a^2)
         S21_2x = S21a^2 / (1 - S22a^2)
     S11a is obtained by time-gating S11_2x to the first half of the thru
     delay, then  S22a = (S11_2x - S11a) / S21_2x  and
     S21a = sqrt(S21_2x (1 - S22a^2))  with the branch fixed by phase
     continuity.  Exact for mirror-symmetric halves once the gate isolates
     the launch.
   * ``symmetric`` (electrically short fixture): each half is additionally
     assumed to be symmetric in itself (S11a = S22a), in which case
     reverse(A) = A and  T_2x = T_a^2, so  T_a = sqrt(T_2x)  (matrix square
     root, branch tracked by continuity in frequency).  Exact for symmetric
     halves; the error equals the half's own asymmetry.

   For balanced-pair fixtures this is done per mode (differential and
   common) and mode conversion inside the fixture is neglected.

3. **Port extension** (:func:`port_extension`): per-port delay and
   sqrt(f) loss only, S'_ij = S_ij L_i L_j exp(+j omega (tau_i + tau_j)).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

from .mixedmode import MixedModeNetwork, PortMap, from_mixed_mode, to_mixed_mode
from .network import Network, reverse_ports, s_to_t, t_to_s
from .tdr import gate

__all__ = ["deembed", "embed", "port_extension", "bisect_2x_thru", "bisect_2x_thru_pair", "thru_delay"]


def _prepare_fixture(fix: Network, meas: Network, side: str) -> np.ndarray:
    if fix.nports != meas.nports:
        raise ValueError(f"{side} fixture has {fix.nports} ports, measurement has {meas.nports}; "
                         "a fixture file must contain both its VNA-side and DUT-side ports")
    fx = fix
    if fx.nfreq != meas.nfreq or not np.allclose(fx.f, meas.f):
        fx = fx.interpolate(meas.f)
    if not np.allclose(fx.z0, meas.z0):
        fx = fx.renormalized(meas.z0)
    return fx.s


def deembed(meas: Network, left: Network | None = None, right: Network | None = None) -> Network:
    """Remove ``left`` and/or ``right`` fixtures from a 2n-port measurement.

    Fixture port convention: ports 0..n-1 = VNA side, n..2n-1 = DUT side.
    """
    if meas.nports % 2:
        raise ValueError("de-embedding needs an even number of ports (left/right groups)")
    t = s_to_t(meas.s)
    if left is not None:
        t = np.linalg.inv(s_to_t(_prepare_fixture(left, meas, "left"))) @ t
    if right is not None:
        t = t @ np.linalg.inv(s_to_t(reverse_ports(_prepare_fixture(right, meas, "right"))))
    out = Network(meas.f, t_to_s(t), meas.z0, name=meas.name, comments=list(meas.comments),
                  port_names=list(meas.port_names))
    out.comments.append("fixture de-embedded (explicit T-matrix method)")
    return out


def embed(dut: Network, left: Network | None = None, right: Network | None = None) -> Network:
    """Inverse of :func:`deembed` (used to build synthetic measurements and tests)."""
    t = s_to_t(dut.s)
    if left is not None:
        t = s_to_t(_prepare_fixture(left, dut, "left")) @ t
    if right is not None:
        t = t @ s_to_t(reverse_ports(_prepare_fixture(right, dut, "right")))
    return Network(dut.f, t_to_s(t), dut.z0, name=dut.name)


def port_extension(net: Network, delays_s, loss_db=None, f_ref: float = 1e9) -> Network:
    """Remove a per-port electrical delay (s) and, optionally, a sqrt(f)
    loss (dB at ``f_ref``) from every port."""
    delays = np.broadcast_to(np.asarray(delays_s, float), (net.nports,))
    w = 2 * np.pi * net.f
    phase = np.exp(1j * w[:, None] * delays[None, :])
    s = net.s * phase[:, :, None] * phase[:, None, :]
    if loss_db is not None:
        loss = np.broadcast_to(np.asarray(loss_db, float), (net.nports,))
        lin = 10 ** (loss[None, :] * np.sqrt(net.f[:, None] / f_ref) / 20)
        s = s * lin[:, :, None] * lin[:, None, :]
    out = net.copy()
    out.s = s
    out.comments.append("port extension applied")
    return out


def thru_delay(f: np.ndarray, s21: np.ndarray) -> float:
    """Mean group delay of a transmission parameter (s)."""
    ph = np.unwrap(np.angle(s21))
    slope = np.polyfit(2 * np.pi * f, ph, 1)[0]
    return float(-slope)


def _bisect_2port(f: np.ndarray, s11: np.ndarray, s21: np.ndarray, gate_fraction: float,
                  beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = thru_delay(f, s21)
    if T <= 0:
        raise ValueError("2x-thru has non-positive delay; check the file")
    s11a = gate(f, s11, t_start=-1.0, t_stop=gate_fraction * T, beta=beta)
    s22a = (s11 - s11a) / s21
    y = s21 * (1 - s22a ** 2)
    ph = np.unwrap(np.angle(y))
    s21a = np.sqrt(np.abs(y)) * np.exp(0.5j * ph)
    return s11a, s21a, s22a


def _sqrt_t_tracked(t: np.ndarray) -> np.ndarray:
    """Matrix square root of a stack of 2x2 wave-chain matrices with the
    branch chosen by continuity in frequency (and Re S21 > 0 at the first point)."""
    lam, v = np.linalg.eig(t)
    vi = np.linalg.inv(v)
    out = np.empty_like(t)
    prev = None
    for k in range(t.shape[0]):
        if abs(lam[k, 0] - lam[k, 1]) < 1e-2:          # near-degenerate: Schur-based principal root
            p = sqrtm(t[k])
            cands = [p, -p]
        else:
            cands = []
            for sgn in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                r = np.sqrt(lam[k]) * np.array(sgn)
                cands.append(v[k] @ np.diag(r) @ vi[k])
        if prev is None:
            best = min(cands, key=lambda c: -(1.0 / c[1, 1]).real)
        else:
            best = min(cands, key=lambda c: np.abs(c - prev).sum())
        out[k] = best
        prev = best
    return out


def _bisect_2port_symmetric(s2x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ta = _sqrt_t_tracked(s_to_t(s2x))
    sa = t_to_s(ta)
    return sa[:, 0, 0], sa[:, 1, 0], sa[:, 1, 1]


def _choose_method(f: np.ndarray, s21: np.ndarray, method: str) -> str:
    if method != "auto":
        return method
    T = thru_delay(f, s21)
    bw = f[-1] - f[0]
    return "gated" if T * bw >= 4.0 else "symmetric"


def _bisect_any(f, s2x, method, gate_fraction, beta):
    m = _choose_method(f, s2x[:, 1, 0], method)
    if m == "gated":
        return (*_bisect_2port(f, s2x[:, 0, 0], s2x[:, 1, 0], gate_fraction, beta), m)
    if m == "symmetric":
        return (*_bisect_2port_symmetric(s2x), m)
    raise ValueError(f"unknown bisection method {method!r}")


def bisect_2x_thru(thru: Network, method: str = "auto", gate_fraction: float = 0.8,
                   beta: float = 6.0) -> Network:
    """Split a symmetric 2-port 2x-thru into its left half (port 0 = VNA side,
    port 1 = DUT side).  The right half is ``reverse_ports`` of the result.

    ``method``: ``"auto"`` (default), ``"gated"`` or ``"symmetric"`` (see module doc)."""
    if thru.nports != 2:
        raise ValueError("bisect_2x_thru expects a 2-port; use bisect_2x_thru_pair for balanced fixtures")
    s11a, s21a, s22a, used = _bisect_any(thru.f, thru.s, method, gate_fraction, beta)
    s = np.zeros_like(thru.s)
    s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1] = s11a, s21a, s21a, s22a
    return Network(thru.f, s, thru.z0, name=thru.name + " (half)",
                   comments=[f"half fixture from 2x-thru bisection ({used})"])


def bisect_2x_thru_pair(thru: Network, method: str = "auto", gate_fraction: float = 0.8,
                        beta: float = 6.0) -> Network:
    """Balanced-pair version: 4-port 2x-thru with ports (1,2) = VNA side (+,-)
    and (3,4) = far side (+,-).  Returns the 4-port left half with ports
    (1,2) VNA side and (3,4) DUT side."""
    if thru.nports != 4:
        raise ValueError("bisect_2x_thru_pair expects a 4-port")
    pm = PortMap.single_pair()
    mm = to_mixed_mode(thru, pm)
    labels = [("d", "A", "near"), ("d", "A", "far"), ("c", "A", "near"), ("c", "A", "far")]
    s = np.zeros((thru.nfreq, 4, 4), dtype=complex)
    used = []
    for mode, (i, j) in (("d", (0, 1)), ("c", (2, 3))):
        a, b = mm.index(mode, "A", "near"), mm.index(mode, "A", "far")
        s2x = mm.s[:, [a, b]][:, :, [a, b]]
        s11a, s21a, s22a, m = _bisect_any(thru.f, s2x, method, gate_fraction, beta)
        used.append(m)
        s[:, i, i], s[:, i, j], s[:, j, i], s[:, j, j] = s11a, s21a, s21a, s22a
    z_mm = np.array([mm.z0[mm.index(*l)] for l in labels])
    half_mm = MixedModeNetwork(thru.f, s, z_mm, labels, name=thru.name + " (half)")
    half = from_mixed_mode(half_mm, pm)
    half.comments = [f"half fixture from per-mode 2x-thru bisection ({'/'.join(used)}); "
                     "fixture mode conversion neglected"]
    return half
