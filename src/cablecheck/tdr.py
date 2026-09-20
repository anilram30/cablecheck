"""
Frequency -> time transforms used for the impedance profile ("TDR from
S-parameters") and for time gating.

Pipeline (one-sided spectrum S(f_k), k = 0..K-1 on a uniform grid of step df)

1. **Uniform grid.** VNA sweeps are normally linear; a non-uniform sweep is
   re-sampled linearly onto a uniform grid of the same span.
2. **DC extrapolation.** A VNA never measures f = 0 but the step response
   needs S(0).  Re S is extrapolated linearly from the first ``n_fit`` points,
   Im S is taken linearly to zero (S(0) of a passive reciprocal network is
   real).  The gap between DC and the first point is filled on the same
   step df.
3. **Windowing.** A Kaiser window w_k (parameter beta) tapers the band
   edge.  The window is the right half of a symmetric window centred on DC so
   that low frequencies are not attenuated.
4. **Rise-time filter (optional).**  A Gaussian low-pass
   H(f) = exp(-ln2 (f/f_3dB)^2) with f_3dB = 0.339/t_r gives the step
   response the 10-90 % rise time t_r that a specification assumes
   (e.g. 500 ps for automotive Ethernet impedance).
5. **Inverse real FFT** with zero padding to n_fft points: the impulse
   response h[n] at t_n = n / (n_fft df).  Because the spectrum is
   Hermitian, h is real.  The step response is the running sum
   r[n] = sum_{m<=n} h[m] started a few resolution cells before t = 0 in
   signed time (the half of an edge at the reference plane that falls at
   negative time wraps to the end of the array and is counted), which
   converges to S(0).
6. **Impedance.** Z(t) = Z_ref (1 + r(t)) / (1 - r(t)); with the round-trip
   time converted to distance x = v t / 2, v = NVP c0.

The transform is invertible enough for time gating: the impulse response is
multiplied by a gate g(t) with raised-cosine edges, transformed back with
rfft, divided by the window (where the window is not too small) and
re-sampled on the original frequency points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import kaiser

C0 = 299_792_458.0

__all__ = ["TimeResponse", "to_time_domain", "impedance_from_step", "gate", "rise_time_filter", "C0"]


@dataclass
class TimeResponse:
    t: np.ndarray          # s
    impulse: np.ndarray    # dimensionless reflection/transmission impulse response (per sample)
    step: np.ndarray       # step response (dimensionless)
    df: float              # frequency step used (Hz)
    n_fft: int
    f_uniform: np.ndarray  # the uniform grid (incl. DC) that was transformed
    s_uniform: np.ndarray  # spectrum on that grid *before* windowing

    def distance(self, nvp: float) -> np.ndarray:
        """One-way distance axis (m) for a reflection response."""
        return nvp * C0 * self.t / 2.0


def _uniform(f: np.ndarray, s: np.ndarray, rel_tol: float = 1e-6):
    df = np.diff(f)
    if np.allclose(df, df[0], rtol=rel_tol):
        return f, s, float(df[0])
    step = float(np.min(df))
    fu = np.arange(f[0], f[-1] + step / 2, step)
    su = np.interp(fu, f, s.real) + 1j * np.interp(fu, f, s.imag)
    return fu, su, step


def extrapolate_dc(f: np.ndarray, s: np.ndarray, n_fit: int = 5):
    """Return (f_ext, s_ext) starting at exactly 0 Hz on the grid step of f."""
    f, s, df = _uniform(f, s)
    k0 = int(round(f[0] / df))
    if abs(k0 * df - f[0]) > 1e-3 * df:
        # first point is not an integer multiple of df: re-grid from DC
        n = int(np.floor(f[-1] / df))
        fu = np.arange(0, n + 1) * df
        m = fu >= f[0]
        su = np.zeros(fu.size, dtype=complex)
        su[m] = np.interp(fu[m], f, s.real) + 1j * np.interp(fu[m], f, s.imag)
        k0 = int(np.argmax(m))
        f, s = fu[m], su[m]
        f_ext = fu
    else:
        f_ext = np.arange(0, k0 + f.size) * df
    n_fit = max(2, min(n_fit, f.size))
    p_re = np.polyfit(f[:n_fit], s[:n_fit].real, 1)
    re_dc = np.polyval(p_re, 0.0)
    # keep |S(0)| physical
    re_dc = float(np.clip(re_dc, -1.0, 1.0))
    f_gap = f_ext[:k0]
    re_gap = re_dc + (s[0].real - re_dc) * (f_gap / f[0]) if k0 else np.array([])
    im_gap = s[0].imag * (f_gap / f[0]) if k0 else np.array([])
    s_ext = np.concatenate([re_gap + 1j * im_gap, s])
    return f_ext, s_ext


def rise_time_filter(f: np.ndarray, t_rise: float) -> np.ndarray:
    """Gaussian low-pass with 10-90 % step rise time ``t_rise`` (seconds)."""
    f3 = 0.339 / t_rise
    return np.exp(-np.log(2.0) * (f / f3) ** 2)


def to_time_domain(f: np.ndarray, s: np.ndarray, beta: float = 6.0, n_fft: int | None = None,
                   t_rise: float | None = None, n_fit: int = 5) -> TimeResponse:
    f_ext, s_ext = extrapolate_dc(np.asarray(f, float), np.asarray(s, complex), n_fit=n_fit)
    df = f_ext[1] - f_ext[0]
    K = f_ext.size
    w = kaiser(2 * K - 1, beta)[K - 1:]  # right half, w[0] = 1 at DC
    x = s_ext * w
    if t_rise is not None:
        x = x * rise_time_filter(f_ext, t_rise)
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(8 * K)))
    h = np.fft.irfft(x, n=n_fft)
    t = np.arange(n_fft) / (n_fft * df)
    # half of an edge at the reference plane sits at negative time, which the
    # DFT wraps to the end of the array; start the running sum a few
    # resolution cells before t = 0 (signed time) so that half is counted
    n_pre = min(int(round(4.0 / f_ext[-1] * n_fft * df)), n_fft // 4)
    step = np.cumsum(h) + (h[n_fft - n_pre:].sum() if n_pre > 0 else 0.0)
    return TimeResponse(t, h, step, df, n_fft, f_ext, s_ext)


def impedance_from_step(step: np.ndarray, z_ref: float, rho_max: float = 0.999) -> np.ndarray:
    r = np.clip(step, -rho_max, rho_max)
    return z_ref * (1 + r) / (1 - r)


def gate(f: np.ndarray, s: np.ndarray, t_start: float, t_stop: float, beta: float = 6.0,
         edge: float | None = None, n_fft: int | None = None, w_min: float = 0.05) -> np.ndarray:
    """Time-gate a reflection/transmission parameter, returned on the input grid.

    The gate is 1 on [t_start, t_stop] with raised-cosine edges of width
    ``edge`` (default: 2/(f span), i.e. two resolution cells) outside it.
    """
    f = np.asarray(f, float)
    tr = to_time_domain(f, s, beta=beta, n_fft=n_fft)
    if edge is None:
        edge = 2.0 / (tr.f_uniform[-1] - tr.f_uniform[0])
    # the impulse response is periodic with period 1/df; a windowed impulse at
    # t = 0 spreads into negative time, which wraps to the end of the array, so
    # use a signed time axis for the gate
    period = 1.0 / tr.df
    t = np.where(tr.t > period / 2, tr.t - period, tr.t)
    g = np.zeros_like(t)
    inside = (t >= t_start) & (t <= t_stop)
    g[inside] = 1.0
    lo = (t < t_start) & (t >= t_start - edge)
    g[lo] = 0.5 * (1 + np.cos(np.pi * (t_start - t[lo]) / edge))
    hi = (t > t_stop) & (t <= t_stop + edge)
    g[hi] = 0.5 * (1 + np.cos(np.pi * (t[hi] - t_stop) / edge))
    hg = tr.impulse * g
    X = np.fft.rfft(hg, n=tr.n_fft)[: tr.f_uniform.size]
    K = tr.f_uniform.size
    w = kaiser(2 * K - 1, beta)[K - 1:]
    corr = np.where(w > w_min, 1.0 / np.maximum(w, w_min), 1.0 / w_min)
    Xg = X * corr
    # back onto the caller's grid
    return np.interp(f, tr.f_uniform, Xg.real) + 1j * np.interp(f, tr.f_uniform, Xg.imag)
