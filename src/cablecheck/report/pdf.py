"""Multi-page PDF report via matplotlib (no LaTeX / external converter needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from ..pipeline import RunResult  # noqa: E402
from .plots import C_FAIL, C_PASS, C_TEXT, TITLES, margin_bar, plot_result  # noqa: E402


def _xfmt(r):
    if r.x_worst is None:
        return "-"
    return f"{r.x_worst / 1e6:.2f} MHz" if r.xunit == "Hz" else f"{r.x_worst:.2f} m"


def write_pdf(result: RunResult, path: str | Path) -> Path:
    path = Path(path)
    ev = result.evaluation
    s = result.sample
    with PdfPages(path) as pdf:
        # page 1: summary
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, f"Cable measurement report - sample {s.sample_id}", fontsize=16, weight="bold")
        fig.text(0.07, 0.925, f"{result.cable_type.title}", fontsize=10, color="#52514e")
        fig.text(0.07, 0.905, f"generated {result.timestamp} by cablecheck {result.software_version}", fontsize=8, color="#52514e")
        col = {"PASS": C_PASS, "FAIL": C_FAIL}.get(ev.verdict, "#52514e")
        fig.text(0.07, 0.86, ev.verdict, fontsize=26, weight="bold", color=col)
        y = 0.83
        h = ev.headline
        if h is not None:
            fig.text(0.07, y, f"Headline: {TITLES.get(h.quantity, h.quantity)} [{h.pair}], worst margin {h.worst_margin:+.2f} {h.unit} "
                     f"at {_xfmt(h)} (measured {h.value_at_worst:.2f}, limit {h.limit_at_worst:.2f})", fontsize=9)
            y -= 0.025
        meta = [("Part number", s.part_number), ("Lot", s.lot), ("Length", f"{s.length_m} m" if s.length_m else ""),
                ("Operator", s.operator), ("Instrument", f"{s.instrument} {s.instrument_serial}".strip()),
                ("Calibration date", s.calibration_date),
                ("Temperature", f"{s.temperature_c} C" if s.temperature_c is not None else ""), ("Site", s.site), ("Notes", s.notes)]
        for k, v in meta:
            if v:
                fig.text(0.07, y, f"{k}: {v}", fontsize=9)
                y -= 0.018
        for fr in result.files:
            fig.text(0.07, y, f"file {fr['name']}  ({fr['nports']} ports, {fr['npoints']} pts, {fr['fmin_hz'] / 1e6:.3g}-{fr['fmax_hz'] / 1e6:.4g} MHz, "
                     f"fixture: {fr['fixture']}, sha256 {fr['sha256'][:12]}...)", fontsize=7.5, color="#52514e")
            y -= 0.016
        y -= 0.01
        # results table
        cols = ["quantity", "pair", "verdict", "worst margin", "at", "measured", "limit", "fail/pts"]
        rows = []
        for r in ev.results:
            word = "info" if r.passed is None else ("PASS" if r.passed else "FAIL")
            rows.append([TITLES.get(r.quantity, r.quantity), r.pair, word,
                         "-" if r.worst_margin is None else f"{r.worst_margin:+.2f} {r.unit}", _xfmt(r),
                         "-" if r.value_at_worst is None else f"{r.value_at_worst:.2f}",
                         "-" if r.limit_at_worst is None else f"{r.limit_at_worst:.2f}", f"{r.n_fail}/{r.n_points}"])
        ax = fig.add_axes([0.05, max(0.05, y - 0.03 - 0.022 * len(rows)), 0.9, 0.022 * len(rows) + 0.03])
        ax.axis("off")
        tbl = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="left",
                       colWidths=[0.28, 0.08, 0.08, 0.14, 0.12, 0.1, 0.1, 0.1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.25)
        for (i, j), cell in tbl.get_celld().items():
            cell.set_edgecolor("#e6e6e3")
            if i == 0:
                cell.set_text_props(weight="bold", color="#52514e")
            elif j == 2:
                cell.set_text_props(color={"PASS": C_PASS, "FAIL": C_FAIL}.get(rows[i - 1][2], C_TEXT), weight="bold")
        pdf.savefig(fig)
        plt.close(fig)
        # page 2: margins
        fig = margin_bar(ev)
        fig.set_size_inches(8.27, max(3, 0.35 * len(ev.limited) + 1.5))
        pdf.savefig(fig)
        plt.close(fig)
        # curves, two per page
        curves = [r for r in ev.results if not r.is_scalar]
        for i in range(0, len(curves), 2):
            fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
            for ax, r in zip(axes, curves[i:i + 2]):
                plot_result(r, ax=ax)
            if len(curves[i:i + 2]) == 1:
                axes[1].axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = f"cablecheck report {s.sample_id}"
        info["Author"] = "cablecheck"
    return path
