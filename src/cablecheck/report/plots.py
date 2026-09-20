"""Matplotlib figures for the report (one style, used by HTML, PDF and GUI)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..evaluate import Evaluation, QuantityResult  # noqa: E402

# palette (colour-vision-safe, from the dataviz reference palette)
C_MEAS = "#2a78d6"     # measured curve
C_LIMIT = "#e34948"    # limit line
C_FAIL = "#e34948"
C_PASS = "#008300"
C_MASK = "#9a9a96"
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"

TITLES = {
    "insertion_loss": "Insertion loss", "return_loss": "Return loss", "lcl": "Mode conversion, near end (LCL)",
    "lctl": "Mode conversion, transfer (LCTL)", "next": "Near-end crosstalk (NEXT)", "fext": "Far-end crosstalk (FEXT)",
    "phase_delay": "Propagation (phase) delay", "group_delay": "Group delay", "delay_per_metre": "Delay per metre",
    "delay_skew": "Delay skew between pairs", "impedance_profile": "Impedance profile (TDR from S_dd11)",
    "impedance_fitted": "Fitted impedance (IEC 61156-1 style)", "impedance_mean": "TDR impedance, window mean",
    "impedance_min": "TDR impedance, window min", "impedance_max": "TDR impedance, window max", "nvp": "Velocity ratio (NVP)",
}


def _style(ax):
    ax.grid(True, which="major", color="#e6e6e3", linewidth=0.8)
    ax.grid(True, which="minor", color="#f2f2f0", linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)


def plot_result(r: QuantityResult, ax=None, log_x: bool = True):
    """Measured curve + limit, worst point marked."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 3.4), dpi=110)
    else:
        fig = ax.figure
    if r.xunit == "Hz":
        x = r.x / 1e6
        xl = "frequency / MHz"
    elif r.xunit == "m":
        x = r.x
        xl = "distance / m"
        log_x = False
    else:
        x = np.arange(r.x.size)
        xl = ""
        log_x = False
    ax.plot(x, r.value, color=C_MEAS, linewidth=1.6, label=f"measured {r.pair}")
    if r.limit is not None:
        m = r.mask
        if r.kind == "range" and r.limit_upper is not None:
            ax.plot(x[m], r.limit[m], color=C_LIMIT, linewidth=1.2, linestyle="--", label="limit")
            ax.plot(x[m], r.limit_upper[m], color=C_LIMIT, linewidth=1.2, linestyle="--")
        else:
            ax.plot(x[m], r.limit[m], color=C_LIMIT, linewidth=1.2, linestyle="--", label="limit")
        if r.margin is not None:
            bad = m & (r.margin < 0)
            if np.any(bad):
                ax.plot(x[bad], r.value[bad], ".", color=C_FAIL, markersize=4, label="fails")
        if r.x_worst is not None and r.worst_margin is not None:
            xw = r.x_worst / 1e6 if r.xunit == "Hz" else r.x_worst
            ax.plot([xw], [r.value_at_worst], "o", markerfacecolor="white", markeredgecolor=C_TEXT,
                    markersize=7, markeredgewidth=1.2, zorder=5)
            ax.annotate(f"worst margin {r.worst_margin:+.2f} {r.unit}",
                        (xw, r.value_at_worst), textcoords="offset points", xytext=(8, 8),
                        fontsize=8, color=C_TEXT)
    if log_x and np.all(x > 0):
        ax.set_xscale("log")
    ax.set_xlabel(xl)
    ax.set_ylabel(r.unit)
    title = TITLES.get(r.quantity, r.quantity)
    tag = "" if r.passed is None else ("  PASS" if r.passed else "  FAIL")
    ax.set_title(f"{title} [{r.pair}]{tag}", fontsize=10, color=(C_PASS if r.passed else C_FAIL) if r.passed is not None else C_TEXT, loc="left")
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    return fig


def fig_to_svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fig_to_png_bytes(fig, dpi=150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def margin_bar(ev: Evaluation, ax=None):
    """Horizontal bar chart of worst margins, one bar per limited quantity."""
    rows = [r for r in ev.limited if r.worst_margin is not None]
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 0.35 * len(rows) + 1.2), dpi=110)
    else:
        fig = ax.figure
    labels = [f"{TITLES.get(r.quantity, r.quantity)} [{r.pair}]" for r in rows]
    vals = [r.worst_margin for r in rows]
    cols = [C_PASS if v >= 0 else C_FAIL for v in vals]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, height=0.6)
    ax.axvline(0, color=C_TEXT, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("worst margin (positive = inside limit); units differ per row")
    for yi, v, r in zip(y, vals, rows):
        ax.text(max(v, 0), yi, f" {v:+.2f} {r.unit}", va="center", ha="left", fontsize=8, color=C_TEXT)
    _style(ax)
    fig.tight_layout()
    return fig
