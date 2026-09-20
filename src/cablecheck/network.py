"""
N-port scattering-parameter container and the linear-algebra conversions
(S <-> Z, reference-impedance renormalisation, wave-chain (T) parameters,
ABCD -> S) that the rest of the package is built on.

Conventions
-----------
* ``f`` is a 1-D array of frequencies in Hz, strictly increasing.
* ``s`` has shape ``(nf, n, n)``; ``s[k, i, j]`` is the response at port *i*
  due to an incident *power wave* at port *j* at frequency ``f[k]``.
* ``z0`` has shape ``(n,)`` and holds the real, positive reference impedance
  of every port.  Power waves at port *i* are
  ``a_i = (V_i + Z0_i I_i) / (2 sqrt(Z0_i))``,
  ``b_i = (V_i - Z0_i I_i) / (2 sqrt(Z0_i))``  (I_i flowing *into* the port).
* Port indices in the public API are **zero based**.

Everything is pure numpy/scipy; no third-party RF library is used so that
every formula in the technical report corresponds to a line of code here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "Network",
    "s_to_z",
    "z_to_s",
    "renormalize",
    "s_to_t",
    "t_to_s",
    "cascade",
    "reverse_ports",
    "abcd_to_s",
    "db",
    "deg",
]


def db(x: np.ndarray) -> np.ndarray:
    """20 log10 |x| with a floor to avoid -inf."""
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))


def deg(x: np.ndarray) -> np.ndarray:
    return np.degrees(np.angle(x))


@dataclass
class Network:
    """An N-port network sampled on a frequency grid."""

    f: np.ndarray
    s: np.ndarray
    z0: np.ndarray
    name: str = ""
    comments: list[str] = field(default_factory=list)
    port_names: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ init
    def __post_init__(self) -> None:
        self.f = np.asarray(self.f, dtype=float).reshape(-1)
        self.s = np.asarray(self.s, dtype=complex)
        if self.s.ndim != 3 or self.s.shape[1] != self.s.shape[2]:
            raise ValueError(f"s must have shape (nf, n, n); got {self.s.shape}")
        if self.s.shape[0] != self.f.size:
            raise ValueError("frequency axis and s-matrix first axis differ")
        n = self.s.shape[1]
        z0 = np.asarray(self.z0, dtype=float).reshape(-1)
        if z0.size == 1:
            z0 = np.full(n, float(z0[0]))
        if z0.size != n:
            raise ValueError("z0 must be scalar or have one entry per port")
        if np.any(z0 <= 0):
            raise ValueError("reference impedances must be positive")
        self.z0 = z0
        if np.any(np.diff(self.f) <= 0):
            raise ValueError("frequencies must be strictly increasing")
        if not self.port_names:
            self.port_names = [f"P{i + 1}" for i in range(n)]

    # ------------------------------------------------------------ properties
    @property
    def nports(self) -> int:
        return self.s.shape[1]

    @property
    def nfreq(self) -> int:
        return self.f.size

    def s_db(self, i: int, j: int) -> np.ndarray:
        return db(self.s[:, i, j])

    def s_deg(self, i: int, j: int) -> np.ndarray:
        return deg(self.s[:, i, j])

    def copy(self) -> "Network":
        return Network(self.f.copy(), self.s.copy(), self.z0.copy(), self.name,
                       list(self.comments), list(self.port_names))

    # -------------------------------------------------------------- slicing
    def subnetwork(self, ports: Sequence[int]) -> "Network":
        """Keep only ``ports`` (zero based), all other ports terminated in
        their reference impedance (which is what dropping rows/columns of S
        means physically)."""
        idx = list(ports)
        s = self.s[:, idx][:, :, idx]
        return Network(self.f, s, self.z0[idx], self.name, list(self.comments),
                       [self.port_names[i] for i in idx])

    def reorder(self, order: Sequence[int]) -> "Network":
        """Renumber ports: new port k is old port ``order[k]``."""
        if sorted(order) != list(range(self.nports)):
            raise ValueError("order must be a permutation of all ports")
        return self.subnetwork(order)

    def crop(self, fmin: float | None = None, fmax: float | None = None) -> "Network":
        m = np.ones(self.nfreq, dtype=bool)
        if fmin is not None:
            m &= self.f >= fmin
        if fmax is not None:
            m &= self.f <= fmax
        return Network(self.f[m], self.s[m], self.z0, self.name, list(self.comments),
                       list(self.port_names))

    def interpolate(self, f_new: np.ndarray) -> "Network":
        """Linear interpolation of real and imaginary parts onto ``f_new``
        (inside the measured span only)."""
        f_new = np.asarray(f_new, dtype=float)
        if f_new.min() < self.f.min() - 1e-9 or f_new.max() > self.f.max() + 1e-9:
            raise ValueError("cannot extrapolate S-parameters")
        n = self.nports
        s = np.empty((f_new.size, n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                s[:, i, j] = (np.interp(f_new, self.f, self.s[:, i, j].real)
                              + 1j * np.interp(f_new, self.f, self.s[:, i, j].imag))
        return Network(f_new, s, self.z0, self.name, list(self.comments), list(self.port_names))

    def renormalized(self, z_new) -> "Network":
        z_new = np.asarray(z_new, dtype=float).reshape(-1)
        if z_new.size == 1:
            z_new = np.full(self.nports, float(z_new[0]))
        return Network(self.f, renormalize(self.s, self.z0, z_new), z_new, self.name,
                       list(self.comments), list(self.port_names))

    # ----------------------------------------------------------- diagnostics
    def passivity_violation(self) -> np.ndarray:
        """Largest singular value of S minus one, per frequency (>0 = active)."""
        return np.linalg.svd(self.s, compute_uv=False)[:, 0] - 1.0

    def reciprocity_error(self) -> np.ndarray:
        """max |S_ij - S_ji| per frequency."""
        return np.max(np.abs(self.s - np.transpose(self.s, (0, 2, 1))), axis=(1, 2))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"Network(name={self.name!r}, nports={self.nports}, "
                f"f=[{self.f[0]:.4g}, {self.f[-1]:.4g}] Hz, nf={self.nfreq}, z0={self.z0})")


# ============================================================ conversions
def _eye_like(s: np.ndarray) -> np.ndarray:
    n = s.shape[-1]
    return np.broadcast_to(np.eye(n, dtype=complex), s.shape)


def s_to_z(s: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Impedance matrix from power-wave S with real reference impedances.

    With G = diag(sqrt(z0)):  Z = G (I + S)(I - S)^{-1} G.
    """
    z0 = np.asarray(z0, dtype=float).reshape(-1)
    g = np.diag(np.sqrt(z0)).astype(complex)
    eye = _eye_like(s)
    inner = np.linalg.solve(eye - s, eye + s)  # (I-S)^{-1}(I+S); commutes with (I+S)
    return g @ inner @ g


def z_to_s(z: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Power-wave S from an impedance matrix (real reference impedances).

    With G = diag(1/sqrt(z0)):  z_n = G Z G,  S = (z_n - I)(z_n + I)^{-1}.
    """
    z0 = np.asarray(z0, dtype=float).reshape(-1)
    g = np.diag(1.0 / np.sqrt(z0)).astype(complex)
    zn = g @ z @ g
    eye = _eye_like(zn)
    # (zn - I)(zn + I)^{-1}  ==  solve((zn+I)^T, (zn-I)^T)^T
    return np.transpose(np.linalg.solve(np.transpose(zn + eye, (0, 2, 1)),
                                        np.transpose(zn - eye, (0, 2, 1))), (0, 2, 1))


def renormalize(s: np.ndarray, z_old, z_new) -> np.ndarray:
    """Change the port reference impedances of an S-matrix (real Z0 only)."""
    z_old = np.asarray(z_old, dtype=float).reshape(-1)
    z_new = np.asarray(z_new, dtype=float).reshape(-1)
    if np.allclose(z_old, z_new):
        return s.copy()
    return z_to_s(s_to_z(s, z_old), z_new)


# --------------------------------------------------------- wave-chain (T)
def _blocks(m: np.ndarray):
    n = m.shape[-1] // 2
    return m[:, :n, :n], m[:, :n, n:], m[:, n:, :n], m[:, n:, n:]


def _assemble(a, b, c, d) -> np.ndarray:
    top = np.concatenate([a, b], axis=2)
    bot = np.concatenate([c, d], axis=2)
    return np.concatenate([top, bot], axis=1)


def s_to_t(s: np.ndarray) -> np.ndarray:
    """Block wave-chain matrix of a 2n-port.

    Ports 0..n-1 form the *left* group, n..2n-1 the *right* group, and
    [b_L ; a_L] = T [a_R ; b_R]  with
        T11 = S12 - S11 S21^{-1} S22,   T12 = S11 S21^{-1},
        T21 = -S21^{-1} S22,            T22 = S21^{-1}.
    Cascading networks is then plain matrix multiplication, T = T_A T_B.
    """
    if s.shape[-1] % 2:
        raise ValueError("T-parameters need an even number of ports")
    s11, s12, s21, s22 = _blocks(s)
    s21_inv = np.linalg.inv(s21)
    t11 = s12 - s11 @ s21_inv @ s22
    t12 = s11 @ s21_inv
    t21 = -s21_inv @ s22
    t22 = s21_inv
    return _assemble(t11, t12, t21, t22)


def t_to_s(t: np.ndarray) -> np.ndarray:
    """Inverse of :func:`s_to_t`:
        S11 = T12 T22^{-1},  S12 = T11 - T12 T22^{-1} T21,
        S21 = T22^{-1},      S22 = -T22^{-1} T21.
    """
    t11, t12, t21, t22 = _blocks(t)
    t22_inv = np.linalg.inv(t22)
    s11 = t12 @ t22_inv
    s12 = t11 - t12 @ t22_inv @ t21
    s21 = t22_inv
    s22 = -t22_inv @ t21
    return _assemble(s11, s12, s21, s22)


def cascade(*s_list: np.ndarray) -> np.ndarray:
    """Cascade 2n-ports left to right (right group of one onto left group of next)."""
    t = s_to_t(s_list[0])
    for s in s_list[1:]:
        t = t @ s_to_t(s)
    return t_to_s(t)


def reverse_ports(s: np.ndarray) -> np.ndarray:
    """Swap the left and right port groups of a 2n-port (view it from the other side)."""
    n = s.shape[-1] // 2
    idx = list(range(n, 2 * n)) + list(range(n))
    return s[:, idx][:, :, idx]


# ------------------------------------------------------------- ABCD -> S
def abcd_to_s(abcd: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """S-matrix of a 2n-port given its chain (ABCD) matrix.

    Chain convention: [V_L ; I_L] = ABCD [V_R ; I_R'] where I_L flows *into*
    the left ports and I_R' flows *out of* the right ports (toward the load).
    Instead of trusting block-matrix identities, the port boundary problem is
    solved directly for 2n unit incident waves, which is exact and cheap.
    """
    nf, n2, _ = abcd.shape
    n = n2 // 2
    z0 = np.asarray(z0, dtype=float).reshape(-1)
    if z0.size == 1:
        z0 = np.full(n2, float(z0[0]))
    zl, zr = z0[:n], z0[n:]
    sq_l, sq_r = np.sqrt(zl), np.sqrt(zr)
    # unknown x = [V_L, I_L, V_R, I_R'] (4n)
    # rows: chain (2n), incident-wave definitions (2n)
    M = np.zeros((nf, 4 * n, 4 * n), dtype=complex)
    rhs = np.zeros((nf, 4 * n, n2), dtype=complex)
    eye = np.eye(2 * n)
    M[:, :2 * n, :2 * n] = eye
    M[:, :2 * n, 2 * n:] = -abcd
    # a_L = (V_L + zl I_L)/(2 sqrt zl)
    for i in range(n):
        M[:, 2 * n + i, i] = 1.0 / (2 * sq_l[i])
        M[:, 2 * n + i, n + i] = zl[i] / (2 * sq_l[i])
        # a_R = (V_R + zr I_R)/(2 sqrt zr) with I_R = -I_R'
        M[:, 3 * n + i, 2 * n + i] = 1.0 / (2 * sq_r[i])
        M[:, 3 * n + i, 3 * n + i] = -zr[i] / (2 * sq_r[i])
    rhs[:, 2 * n:, :] = np.eye(n2)
    x = np.linalg.solve(M, rhs)  # (nf, 4n, 2n)
    vl, il, vr, irp = x[:, :n], x[:, n:2 * n], x[:, 2 * n:3 * n], x[:, 3 * n:]
    bl = (vl - zl[None, :, None] * il) / (2 * sq_l[None, :, None])
    br = (vr + zr[None, :, None] * irp) / (2 * sq_r[None, :, None])
    return np.concatenate([bl, br], axis=1)


def check_grid_equal(nets: Iterable[Network]) -> np.ndarray:
    nets = list(nets)
    f = nets[0].f
    for n in nets[1:]:
        if n.f.shape != f.shape or not np.allclose(n.f, f):
            raise ValueError(f"frequency grids differ between {nets[0].name!r} and {n.name!r}; "
                             "interpolate first")
    return f
