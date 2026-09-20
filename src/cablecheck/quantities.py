"""
Derived cable quantities from a mixed-mode network.

All "loss" quantities are returned as positive dB:  X = -20 log10 |S_...|,
so that a *higher* number is always *better* for return loss, mode
conversion and crosstalk, and *lower* is better for insertion loss.

Quantity                Symbol / definition                          better
----------------------  -------------------------------------------  ------
insertion_loss          IL = -20 log10 |S_dd21|                      lower
return_loss             RL = min_end -20 log10 |S_dd,ii|             higher
lcl  (mode conv., NE)   min over ends of -20 log10 |S_cd,ii|,|S_dc,ii| higher
lctl (mode conv., FE)   min over dirs of -20 log10 |S_cd,ji|,|S_dc,ji| higher
next                    NEXT = -20 log10 |S_dd(Y near <- X near)|     higher
fext                    FEXT = -20 log10 |S_dd(Y far  <- X near)|     higher
phase_delay             tau_p = -phi_abs(S_dd21) / (2 pi f)           lower
group_delay             tau_g = -d phi / d omega                      -
delay_skew              max_pair tau_p - min_pair tau_p               lower
impedance_profile       Z(x) from TDR of S_dd11 at rise time t_r      range
impedance_mean/min/max  statistics of Z(x) over a near-end window     range
                        [mask, mask + window]; a lossy line's TDR
                        trace rises with distance (loss-compensated
                        profile: see the impedance-profile project)
impedance_fitted        IEC 61156-1 style fitted impedance: |Z_in(f)| =
                        |Z_ref (1+S_dd11)/(1-S_dd11)| smoothed over a
                        frequency window that averages the length
                        ripple, fitted to a + b/sqrt(f), band mean     range

The absolute phase phi_abs needed for the phase delay is recovered from
the unwrapped phase by fixing the integer number of 2 pi turns at the first
frequency point so that the phase delay agrees with the low-frequency group
delay (a cable is a pure delay line at low frequency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .mixedmode import PortMap, to_mixed_mode
from .network import Network, db
from .tdr import C0, impedance_from_step, to_time_domain

__all__ = ["Trace", "compute_quantities", "phase_delay", "group_delay", "impedance_profile"]


@dataclass
class Trace:
    """A quantity as a function of an independent variable (frequency in Hz,
    distance in m, or a single scalar with ``x = [nan]``)."""

    quantity: str          # canonical name, e.g. "insertion_loss"
    pair: str              # "A", "A->B", "all"
    x: np.ndarray
    y: np.ndarray
    unit: str              # "dB", "ns", "ohm", "ns/m"
    xunit: str = "Hz"      # "Hz", "m", ""
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.quantity}[{self.pair}]"

    @property
    def is_scalar(self) -> bool:
        return self.x.size == 1 and self.xunit == ""


# ----------------------------------------------------------------- delays
def _phase_unwrapped(s21: np.ndarray, f: np.ndarray) -> np.ndarray:
    ph = np.unwrap(np.angle(s21))
    # check the grid is fine enough for unambiguous unwrapping
    dphi = np.abs(np.diff(ph))
    if dphi.size and np.max(dphi) > 0.9 * np.pi:
        raise ValueError("frequency grid too coarse to unwrap the transmission phase; "
                         "increase the number of sweep points")
    return ph


def group_delay(f: np.ndarray, s21: np.ndarray) -> np.ndarray:
    ph = _phase_unwrapped(s21, f)
    return -np.gradient(ph, 2 * np.pi * f)


def phase_delay(f: np.ndarray, s21: np.ndarray) -> np.ndarray:
    ph = _phase_unwrapped(s21, f)
    n_low = max(3, f.size // 10)
    slope = np.polyfit(2 * np.pi * f[:n_low], ph[:n_low], 1)[0]
    tau_g_low = -slope
    # phi_abs(f0) should be about -2 pi f0 tau_g;  ph[0] differs by 2 pi k
    k = np.round((ph[0] + 2 * np.pi * f[0] * tau_g_low) / (2 * np.pi))
    ph_abs = ph - 2 * np.pi * k
    return -ph_abs / (2 * np.pi * f)


# ------------------------------------------------------------ impedance
def impedance_profile(f: np.ndarray, s11: np.ndarray, z_ref: float, t_rise: float = 500e-12,
                      nvp: float = 0.67, beta: float = 6.0, x_max: float | None = None):
    """Impedance versus one-way distance from a reflection parameter."""
    tr = to_time_domain(f, s11, beta=beta, t_rise=t_rise)
    z = impedance_from_step(tr.step, z_ref)
    x = tr.distance(nvp)
    if x_max is not None:
        m = x <= x_max
        x, z = x[m], z[m]
    return x, z, tr


def fitted_impedance(f: np.ndarray, s11: np.ndarray, z_ref: float, fmin: float = 10e6,
                     smooth_hz: float = 30e6) -> tuple[float, float, float]:
    """IEC 61156-1 style fitted impedance.  Returns (a, b, band mean of a + b/sqrt(f_MHz))."""
    zin = np.abs(z_ref * (1 + s11) / (1 - s11))
    df = float(np.median(np.diff(f)))
    n = max(1, int(round(smooth_hz / df)))
    zs = np.convolve(zin, np.ones(n) / n, mode="same")
    idx = np.arange(f.size)
    m = (f >= fmin) & (idx >= n // 2) & (idx < f.size - n // 2)
    if m.sum() < 4:
        m = f >= f[0]
        zs = zin
    A = np.vstack([np.ones(m.sum()), 1 / np.sqrt(f[m] / 1e6)]).T
    a, b = np.linalg.lstsq(A, zs[m], rcond=None)[0]
    zfit = a + b / np.sqrt(f[m] / 1e6)
    return float(a), float(b), float(zfit.mean())


def _stats_region(x: np.ndarray, z: np.ndarray, x0: float, x1: float):
    m = (x >= x0) & (x <= x1)
    if not np.any(m):
        return np.nan, np.nan, np.nan
    zz = z[m]
    return float(np.mean(zz)), float(np.min(zz)), float(np.max(zz))


# ------------------------------------------------------------ main entry
def compute_quantities(net: Network, pmap: PortMap, length_m: float | None = None,
                       t_rise: float = 500e-12, nvp: float | None = None,
                       connector_mask_m: float = 0.5, impedance_window_m: float = 3.0,
                       want: Iterable[str] | None = None, profile_engine: str = "plain") -> list[Trace]:
    """Compute every quantity that the port map allows.

    Parameters
    ----------
    length_m
        physical cable length; enables per-metre delay and the distance
        window for the impedance statistics.
    nvp
        nominal velocity of propagation (fraction of c0); if None it is
        estimated from the phase delay and ``length_m`` (falls back to 0.67).
    connector_mask_m
        distance at the near end excluded from the impedance statistics
        (connector/launch region).
    impedance_window_m
        length of the near-end window over which the TDR impedance
        statistics are taken.
    profile_engine
        ``"plain"``: the reflection-coefficient TDR transform of this package;
        ``"zprofile"``: the loss-aware reconstruction of the companion
        package (Project D), if installed - the impedance statistics are then
        taken over the whole un-masked cable instead of a near-end window.
    """
    mm = to_mixed_mode(net, pmap)
    f = net.f
    out: list[Trace] = []
    want = set(want) if want is not None else None

    def wanted(q: str) -> bool:
        return want is None or q in want

    pairs = pmap.pairs
    delays: dict[str, np.ndarray] = {}
    for p in pairs:
        ends = [e for e in ("near", "far") if (p, e) in pmap.groups()]
        if len(ends) == 2:
            s21 = mm.param(("d", p, "far"), ("d", p, "near"))
            s12 = mm.param(("d", p, "near"), ("d", p, "far"))
            if wanted("insertion_loss"):
                out.append(Trace("insertion_loss", p, f, -db(s21), "dB",
                                 meta={"reverse_db": (-db(s12)).tolist()}))
            # mode conversion transfer (both directions, both conversions)
            if wanted("lctl"):
                cands = [mm.param(("c", p, "far"), ("d", p, "near")), mm.param(("d", p, "far"), ("c", p, "near")),
                         mm.param(("c", p, "near"), ("d", p, "far")), mm.param(("d", p, "near"), ("c", p, "far"))]
                out.append(Trace("lctl", p, f, np.min([-db(c) for c in cands], axis=0), "dB"))
            tp = phase_delay(f, s21)
            tg = group_delay(f, s21)
            delays[p] = tp
            if wanted("phase_delay"):
                out.append(Trace("phase_delay", p, f, tp * 1e9, "ns"))
            if wanted("group_delay"):
                out.append(Trace("group_delay", p, f, tg * 1e9, "ns"))
            if length_m:
                if wanted("delay_per_metre"):
                    out.append(Trace("delay_per_metre", p, f, tp * 1e9 / length_m, "ns/m"))
                if wanted("nvp"):
                    # NVP from the phase delay at the upper end of the band
                    tau = float(np.median(tp[-max(5, f.size // 20):]))
                    out.append(Trace("nvp", p, np.array([np.nan]), np.array([length_m / (C0 * tau)]), "", ""))
        # per-end quantities
        rl_ends, lcl_ends = [], []
        for e in ends:
            sdd = mm.param(("d", p, e), ("d", p, e))
            rl_ends.append(-db(sdd))
            lcl_ends.append(np.minimum(-db(mm.param(("c", p, e), ("d", p, e))),
                                       -db(mm.param(("d", p, e), ("c", p, e)))))
        if ends and wanted("return_loss"):
            out.append(Trace("return_loss", p, f, np.min(rl_ends, axis=0), "dB",
                             meta={"ends": ends, "per_end_db": [r.tolist() for r in rl_ends]}))
        if ends and wanted("lcl"):
            out.append(Trace("lcl", p, f, np.min(lcl_ends, axis=0), "dB"))
        # impedance profile from the near end (or whichever end exists)
        if ends and wanted("impedance_profile") and profile_engine == "zprofile":
            try:
                from zprofile import Settings as _ZS
                from zprofile import compute_profile as _zp
            except ImportError as exc:  # pragma: no cover
                raise ImportError("profile_engine='zprofile' needs the zprofile package installed") from exc
            e = ends[0]
            sdd = mm.param(("d", p, e), ("d", p, e))
            z_ref = float(mm.z0[mm.index("d", p, e)])
            other = "far" if e == "near" else "near"
            kw = {}
            if len(ends) == 2:
                kw = dict(s21=mm.param(("d", p, other), ("d", p, e)), s12=mm.param(("d", p, e), ("d", p, other)),
                          s22=mm.param(("d", p, other), ("d", p, other)))
            prof = _zp(f, sdd, z_ref, length_m, settings=_ZS(t_rise=t_rise, x_min=connector_mask_m,
                                                                x_end_margin=connector_mask_m), **kw)
            out.append(Trace("impedance_profile", p, prof.x, prof.z, "ohm", "m",
                             meta={"engine": "zprofile", "nvp": prof.velocity / C0, "t_rise_ps": t_rise * 1e12,
                                   "z_ref": z_ref, "end": e, "resolution_m": prof.resolution_m,
                                   "mask": [prof.mask.x_start, prof.mask.x_end], "features": prof.features}))
            st = prof.stats
            for q, v in (("impedance_mean", st.get("mean", np.nan)), ("impedance_min", st.get("min", np.nan)),
                         ("impedance_max", st.get("max", np.nan))):
                if wanted(q):
                    out.append(Trace(q, p, np.array([np.nan]), np.array([v]), "ohm", "",
                                     meta={"window_m": st.get("window_m"), "engine": "zprofile"}))
        elif ends and wanted("impedance_profile"):
            e = ends[0]
            sdd = mm.param(("d", p, e), ("d", p, e))
            z_ref = float(mm.z0[mm.index("d", p, e)])
            nvp_use = nvp
            if nvp_use is None:
                if length_m and p in delays:
                    tau = float(np.median(delays[p][-max(5, f.size // 20):]))
                    nvp_use = float(np.clip(length_m / (C0 * tau), 0.3, 1.0))
                else:
                    nvp_use = 0.67
            x_max = (length_m * 1.3 + 1.0) if length_m else None
            x, z, _ = impedance_profile(f, sdd, z_ref, t_rise=t_rise, nvp=nvp_use, x_max=x_max)
            out.append(Trace("impedance_profile", p, x, z, "ohm", "m",
                             meta={"nvp": nvp_use, "t_rise_ps": t_rise * 1e12, "z_ref": z_ref,
                                   "end": e}))
            x0 = connector_mask_m
            x1 = x0 + impedance_window_m
            if length_m:
                x1 = min(x1, max(x0 + 0.1, length_m - connector_mask_m))
            zm, zmin, zmax = _stats_region(x, z, x0, x1)
            for q, v in (("impedance_mean", zm), ("impedance_min", zmin), ("impedance_max", zmax)):
                if wanted(q):
                    out.append(Trace(q, p, np.array([np.nan]), np.array([v]), "ohm", "",
                                     meta={"window_m": [x0, x1]}))
        if ends and wanted("impedance_fitted"):
            e = ends[0]
            sdd = mm.param(("d", p, e), ("d", p, e))
            z_ref = float(mm.z0[mm.index("d", p, e)])
            a, b, zmean = fitted_impedance(f, sdd, z_ref)
            out.append(Trace("impedance_fitted", p, np.array([np.nan]), np.array([zmean]), "ohm", "",
                             meta={"a": a, "b": b, "model": "a + b/sqrt(f_MHz)", "end": e}))
    # crosstalk between pairs
    for i, pa in enumerate(pairs):
        for pb in pairs[i + 1:]:
            key = f"{pa}->{pb}"
            if (pa, "near") in pmap.groups() and (pb, "near") in pmap.groups() and wanted("next"):
                n1 = -db(mm.param(("d", pb, "near"), ("d", pa, "near")))
                n2 = -db(mm.param(("d", pa, "near"), ("d", pb, "near")))
                out.append(Trace("next", key, f, np.minimum(n1, n2), "dB"))
            if (pa, "far") in pmap.groups() and (pb, "far") in pmap.groups() and wanted("next"):
                n1 = -db(mm.param(("d", pb, "far"), ("d", pa, "far")))
                n2 = -db(mm.param(("d", pa, "far"), ("d", pb, "far")))
                out.append(Trace("next", key + " (far end)", f, np.minimum(n1, n2), "dB"))
            if ((pa, "near") in pmap.groups() and (pb, "far") in pmap.groups()) and wanted("fext"):
                cands = [-db(mm.param(("d", pb, "far"), ("d", pa, "near")))]
                if (pb, "near") in pmap.groups() and (pa, "far") in pmap.groups():
                    cands.append(-db(mm.param(("d", pa, "far"), ("d", pb, "near"))))
                out.append(Trace("fext", key, f, np.min(cands, axis=0), "dB"))
    if len(delays) >= 2 and wanted("delay_skew"):
        stack = np.array(list(delays.values())) * 1e9
        out.append(Trace("delay_skew", "all", f, stack.max(axis=0) - stack.min(axis=0), "ns"))
    return out
