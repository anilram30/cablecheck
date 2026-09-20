"""
Physics-based synthetic cables, fixtures and "measurements".

The lab's real Touchstone files are confidential, so the reference data set
that the regression tests run on is *generated* from multiconductor
transmission-line (MTL) theory.  Nothing here is fitted to a real cable; the
point is that every number the pipeline produces on this data can be traced
back to a closed-form model, which is what makes it a *reference*.

Model
-----
A pair is two conductors above a reference (the shield).  Per unit length,
with 2x2 matrices,

    Z(f) = R(f) + j omega L,     Y(f) = G(f) + j omega C,

    R(f) = diag(R_dc + R_s sqrt(f)) * (proximity factor),   R_s = sqrt(mu0 rho / pi) / d
    G(f) = omega tan(delta) C.

L and C are built from the odd/even-mode impedances and velocities:

    L_odd = Z_odd / v_odd,  C_odd = 1 / (Z_odd v_odd)    (Z_diff = 2 Z_odd)
    L_ev  = Z_ev  / v_ev,   C_ev  = 1 / (Z_ev  v_ev)     (Z_comm = Z_ev / 2)
    L = [[Ls, Lm], [Lm, Ls]],  Ls = (L_odd + L_ev)/2,  Lm = (L_ev - L_odd)/2
    C = [[Cs, -Cm], [-Cm, Cs]], Cs = (C_odd + C_ev)/2,  Cm = (C_odd - C_ev)/2

Asymmetry (mode conversion) is introduced by scaling conductor 2's
self-capacitance and resistance; longitudinal non-uniformity (impedance
ripple, local defects) by cascading short uniform segments whose C is
scaled by (1 + delta(z)).  Each segment's chain matrix is

    Phi_k = expm( l_k [[0, Z], [Y, 0]] )

with [V_left; I_left] = Phi [V_right; I_right'] (I_right' toward the load),
evaluated by eigendecomposition, and the cable's chain matrix is the
product of the segment matrices.  The S-matrix follows from
:func:`cablecheck.network.abcd_to_s`.

Two coupled pairs (8 ports) use a 4x4 version with inter-pair mutual L and C
that is deliberately *unequal* for the two wires of a pair, which is what
produces differential crosstalk in a real cable.

Fixtures are one launch line plus lumped connector parasitics per wire.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deembed import embed
from .network import Network, abcd_to_s

MU0 = 4e-7 * np.pi
RHO_CU = 1.72e-8
C0 = 299_792_458.0

__all__ = ["PairSpec", "FixtureSpec", "make_pair", "make_two_pairs", "make_fixture",
           "synthesize_measurement", "expm_stack", "chain_line"]


@dataclass
class PairSpec:
    length_m: float = 15.0
    z_diff: float = 100.0
    z_comm: float = 35.0
    nvp: float = 0.68            # odd-mode velocity / c0
    nvp_even: float = 0.66
    d_wire_m: float = 0.5e-3     # conductor diameter
    r_dc_ohm_per_m: float = 0.09
    tan_delta: float = 0.0015
    proximity: float = 1.15      # multiplies the ac resistance
    asym_c: float = 0.0          # fractional capacitance asymmetry of wire 2
    asym_r: float = 0.0          # fractional resistance asymmetry of wire 2
    ripple_amp: float = 0.0      # fractional periodic C ripple (capstan signature)
    ripple_period_m: float = 0.3
    ripple_phase: float = 0.0
    defect_pos_m: float | None = None   # local impedance defect
    defect_amp: float = 0.0             # fractional C bump at the defect
    defect_width_m: float = 0.05
    roughness: float = 0.0       # random C roughness (fraction, rms)
    n_segments: int = 60
    seed: int = 0

    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        """(segment lengths, fractional C deviation per segment)."""
        n = max(1, self.n_segments)
        edges = np.linspace(0, self.length_m, n + 1)
        z = 0.5 * (edges[:-1] + edges[1:])
        delta = self.ripple_amp * np.sin(2 * np.pi * z / self.ripple_period_m + self.ripple_phase)
        if self.defect_pos_m is not None and self.defect_amp:
            delta = delta + self.defect_amp * np.exp(-0.5 * ((z - self.defect_pos_m) / self.defect_width_m) ** 2)
        if self.roughness:
            rng = np.random.default_rng(self.seed)
            delta = delta + self.roughness * rng.standard_normal(n)
        return np.diff(edges), delta


@dataclass
class FixtureSpec:
    launch_len_m: float = 0.03
    launch_z: float = 52.0
    launch_nvp: float = 0.6
    launch_loss_db_per_m_at_1ghz: float = 3.0
    conn_series_l: float = 0.6e-9
    conn_shunt_c: float = 0.25e-12
    # small imbalance between the two wires of the fixture
    wire2_series_l_scale: float = 1.0


# ---------------------------------------------------------------- helpers
def expm_stack(m: np.ndarray, length: float) -> np.ndarray:
    """expm(length * m) for a stack of (nf, k, k) matrices via eigendecomposition."""
    w, v = np.linalg.eig(m)
    e = np.exp(length * w)
    return v @ (e[:, :, None] * np.linalg.inv(v))


def chain_line(z: np.ndarray, y: np.ndarray, length: float) -> np.ndarray:
    """Chain matrix of a uniform MTL segment from per-unit-length Z, Y (nf, n, n)."""
    nf, n, _ = z.shape
    m = np.zeros((nf, 2 * n, 2 * n), dtype=complex)
    m[:, :n, n:] = z
    m[:, n:, :n] = y
    return expm_stack(m, length)


def _lc_from_modes(z_odd, v_odd, z_ev, v_ev):
    l_odd, c_odd = z_odd / v_odd, 1 / (z_odd * v_odd)
    l_ev, c_ev = z_ev / v_ev, 1 / (z_ev * v_ev)
    ls, lm = (l_odd + l_ev) / 2, (l_ev - l_odd) / 2
    cs, cm = (c_odd + c_ev) / 2, (c_odd - c_ev) / 2
    L = np.array([[ls, lm], [lm, ls]])
    C = np.array([[cs, -cm], [-cm, cs]])
    return L, C


def _pair_rlgc(spec: PairSpec, f: np.ndarray):
    L, C = _lc_from_modes(spec.z_diff / 2, spec.nvp * C0, spec.z_comm * 2, spec.nvp_even * C0)
    r_s = np.sqrt(MU0 * RHO_CU / np.pi) / spec.d_wire_m
    r = (spec.r_dc_ohm_per_m + r_s * np.sqrt(f)) * spec.proximity
    R = np.zeros((f.size, 2, 2))
    R[:, 0, 0] = r
    R[:, 1, 1] = r * (1 + spec.asym_r)
    C2 = C.copy()
    C2[1, 1] *= (1 + spec.asym_c)
    return R, L, C2


def _s_from_segments(f, R, L, C, seg_len, delta, tan_delta, z0=50.0):
    """Cascade uniform segments; G = omega tan(delta) C so Y = j omega C (1 - j tan delta)."""
    w = 2 * np.pi * f
    n = L.shape[0]
    Z = R + 1j * w[:, None, None] * L[None]
    phi = None
    for lk, dk in zip(seg_len, delta):
        Ck = C * (1 + dk)
        Y = (1j * w[:, None, None] * Ck[None]) * (1 - 1j * tan_delta)
        p = chain_line(Z, Y, lk)
        phi = p if phi is None else phi @ p
    return abcd_to_s(phi, np.full(2 * n, z0))


def make_pair(spec: PairSpec, f: np.ndarray, z0: float = 50.0, name: str = "pair") -> Network:
    """4-port S-parameters of one pair: ports 1,2 = near (+,-), 3,4 = far (+,-)."""
    f = np.asarray(f, float)
    R, L, C = _pair_rlgc(spec, f)
    seg_len, delta = spec.segments()
    s = _s_from_segments(f, R, L, C, seg_len, delta, spec.tan_delta, z0)
    return Network(f, s, np.full(4, z0), name=name,
                   comments=[f"synthetic pair, length {spec.length_m} m"])


def make_two_pairs(spec_a: PairSpec, spec_b: PairSpec, f: np.ndarray, coupling_l: float = 0.05,
                   coupling_c: float = 0.04, imbalance: float = 0.35, z0: float = 50.0,
                   name: str = "twopair") -> Network:
    """8-port: pair A ports 1-4 (near +,-, far +,-), pair B ports 5-8.

    ``coupling_l``/``coupling_c`` are the inter-pair mutual terms relative to
    the self terms; ``imbalance`` is the fractional difference between the
    coupling of wire a1 to b1 and a1 to b2 (what produces differential crosstalk).
    """
    f = np.asarray(f, float)
    Ra, La, Ca = _pair_rlgc(spec_a, f)
    Rb, Lb, Cb = _pair_rlgc(spec_b, f)
    L = np.zeros((4, 4))
    C = np.zeros((4, 4))
    L[:2, :2], L[2:, 2:] = La, Lb
    C[:2, :2], C[2:, 2:] = Ca, Cb
    ls = 0.5 * (La[0, 0] + Lb[0, 0])
    cs = 0.5 * (Ca[0, 0] + Cb[0, 0])
    kl = coupling_l * ls * np.array([[1 + imbalance, 1 - imbalance], [1 - imbalance, 1 + imbalance]])
    kc = coupling_c * cs * np.array([[1 + imbalance, 1 - imbalance], [1 - imbalance, 1 + imbalance]])
    L[:2, 2:], L[2:, :2] = kl, kl.T
    C[:2, 2:], C[2:, :2] = -kc, -kc.T
    # Maxwell form: added mutual capacitance also appears on the diagonals
    C[0, 0] += kc[0].sum()
    C[1, 1] += kc[1].sum()
    C[2, 2] += kc[:, 0].sum()
    C[3, 3] += kc[:, 1].sum()
    R = np.zeros((f.size, 4, 4))
    R[:, :2, :2], R[:, 2:, 2:] = Ra, Rb
    seg_len, delta = spec_a.segments()
    s = _s_from_segments(f, R, L, C, seg_len, delta, spec_a.tan_delta, z0)
    # chain ordering is [near a1 a2 b1 b2 | far a1 a2 b1 b2]; reorder to A(1-4) B(5-8)
    order = [0, 1, 4, 5, 2, 3, 6, 7]
    s = s[:, order][:, :, order]
    return Network(f, s, np.full(8, z0), name=name, comments=["synthetic two coupled pairs"])


def make_fixture(spec: FixtureSpec, f: np.ndarray, z0: float = 50.0, name: str = "fixture") -> Network:
    """4-port fixture: ports 1,2 = VNA side (+,-), 3,4 = DUT side (+,-).
    VNA -> launch line -> series L -> shunt C -> DUT, per wire, uncoupled."""
    f = np.asarray(f, float)
    w = 2 * np.pi * f
    nf = f.size
    v = spec.launch_nvp * C0
    l_pul, c_pul = spec.launch_z / v, 1 / (spec.launch_z * v)
    # loss: alpha [Np/m] = loss_dB/8.686 * sqrt(f/1GHz)  ->  R = 2 alpha Z
    alpha = spec.launch_loss_db_per_m_at_1ghz / 8.686 * np.sqrt(f / 1e9)
    r_pul = 2 * alpha * spec.launch_z
    eye = np.eye(2)
    Z = (r_pul + 1j * w * l_pul)[:, None, None] * eye[None]
    Y = (1j * w * c_pul)[:, None, None] * eye[None]
    line = chain_line(Z, Y, spec.launch_len_m)
    ser = np.zeros((nf, 4, 4), dtype=complex)
    ser[:, :, :] = np.eye(4)
    ser[:, 0, 2] = 1j * w * spec.conn_series_l
    ser[:, 1, 3] = 1j * w * spec.conn_series_l * spec.wire2_series_l_scale
    sh = np.zeros((nf, 4, 4), dtype=complex)
    sh[:, :, :] = np.eye(4)
    sh[:, 2, 0] = 1j * w * spec.conn_shunt_c
    sh[:, 3, 1] = 1j * w * spec.conn_shunt_c
    abcd = line @ ser @ sh
    s = abcd_to_s(abcd, np.full(4, z0))
    return Network(f, s, np.full(4, z0), name=name, comments=["synthetic launch + connector fixture"])


def synthesize_measurement(dut: Network, fixture: Network | None, noise_db: float = -85.0,
                           seed: int = 1, name: str | None = None) -> Network:
    """Embed the DUT between two copies of the fixture and add measurement noise."""
    meas = dut if fixture is None else embed(dut, fixture, fixture)
    rng = np.random.default_rng(seed)
    sigma = 10 ** (noise_db / 20) / np.sqrt(2)
    noise = sigma * (rng.standard_normal(meas.s.shape) + 1j * rng.standard_normal(meas.s.shape))
    s = meas.s + noise
    return Network(dut.f, s, dut.z0, name=name or dut.name,
                   comments=[f"synthetic measurement, noise {noise_db} dB"])
