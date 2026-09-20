"""Self-contained HTML report (inline SVG figures, no external assets)."""
from __future__ import annotations

import html
from pathlib import Path

from ..pipeline import RunResult
from .plots import TITLES, fig_to_svg, margin_bar, plot_result

_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#0b0b0b;background:#fcfcfb;margin:0;padding:24px;max-width:1100px;margin:auto}
h1{font-size:22px;margin:0 0 4px 0} h2{font-size:16px;margin:28px 0 8px 0;border-bottom:1px solid #e6e6e3;padding-bottom:4px}
.verdict{display:inline-block;font-size:28px;font-weight:700;padding:6px 18px;border-radius:6px;color:#fff;margin:8px 0}
.pass{background:#008300}.fail{background:#e34948}.none{background:#52514e}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid #e6e6e3}
th{color:#52514e;font-weight:600;font-size:12px}td.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{font-weight:700}.tag.pass{color:#008300;background:none}.tag.fail{color:#e34948;background:none}.tag.info{color:#52514e;background:none}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 24px;font-size:13px}
.meta div span{color:#52514e}
.fig{margin:10px 0}.fig svg{max-width:100%;height:auto}
.warn{background:#fff6e5;border-left:4px solid #eda100;padding:8px 12px;font-size:13px;margin:6px 0}
.small{font-size:12px;color:#52514e}
"""


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _fmt(v, nd=2):
    return "–" if v is None else f"{v:.{nd}f}"


def _measured(r) -> str:
    if r.value_at_worst is not None:
        return f"{r.value_at_worst:.2f} {_esc(r.unit)}"
    if r.is_scalar:
        return f"{float(r.value[0]):.3f} {_esc(r.unit)}"
    return "–"


def _xfmt(r):
    if r.x_worst is None:
        return "–"
    if r.xunit == "Hz":
        return f"{r.x_worst / 1e6:.2f} MHz"
    if r.xunit == "m":
        return f"{r.x_worst:.2f} m"
    return ""


def render_html(result: RunResult, embed_figures: bool = True) -> str:
    ev = result.evaluation
    s = result.sample
    ct = result.cable_type
    cls = {"PASS": "pass", "FAIL": "fail"}.get(ev.verdict, "none")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>cablecheck report {_esc(s.sample_id)}</title>"
             f"<style>{_CSS}</style></head><body>"]
    parts.append(f"<h1>Cable measurement report — sample {_esc(s.sample_id)}</h1>")
    parts.append(f"<div class='small'>{_esc(ct.title)} &middot; generated {_esc(result.timestamp)} by cablecheck "
                 f"{_esc(result.software_version)}</div>")
    parts.append(f"<div class='verdict {cls}'>{_esc(ev.verdict)}</div>")
    h = ev.headline
    if h is not None:
        parts.append(f"<div>Headline: <b>{_esc(TITLES.get(h.quantity, h.quantity))} [{_esc(h.pair)}]</b>, worst margin "
                     f"<b>{h.worst_margin:+.2f} {_esc(h.unit)}</b> at {_esc(_xfmt(h))} "
                     f"(measured {_fmt(h.value_at_worst)} {_esc(h.unit)}, limit {_fmt(h.limit_at_worst)} {_esc(h.unit)})</div>")
    # metadata
    parts.append("<h2>Sample and measurement context</h2><div class='meta'>")
    for k, v in [("Sample", s.sample_id), ("Part number", s.part_number), ("Lot", s.lot), ("Length", f"{s.length_m} m" if s.length_m else ""),
                 ("Cable type / limits", f"{ct.id} ({ct.status})"), ("Operator", s.operator), ("Instrument", s.instrument),
                 ("Instrument serial", s.instrument_serial), ("Calibration date", s.calibration_date),
                 ("Temperature", f"{s.temperature_c} °C" if s.temperature_c is not None else ""),
                 ("Humidity", f"{s.humidity_pct} %" if s.humidity_pct is not None else ""), ("Site", s.site), ("Notes", s.notes)]:
        if v:
            parts.append(f"<div><span>{_esc(k)}:</span> {_esc(v)}</div>")
    parts.append("</div>")
    parts.append("<h3 class='small'>Source files</h3><table><tr><th>file</th><th>ports</th><th>span</th><th>points</th><th>port map</th><th>fixture</th><th>sha256</th></tr>")
    for fr in result.files:
        parts.append(f"<tr><td>{_esc(fr['name'])}</td><td>{fr['nports']}</td><td>{fr['fmin_hz'] / 1e6:.3g}–{fr['fmax_hz'] / 1e6:.4g} MHz</td>"
                     f"<td>{fr['npoints']}</td><td>{_esc(fr['port_map'])}</td><td>{_esc(fr['fixture'])}</td><td class='small'>{fr['sha256'][:16]}…</td></tr>")
    parts.append("</table>")
    warns = result.warnings + ev.warnings
    if warns:
        parts.append("<h2>Warnings</h2>")
        for w in warns:
            parts.append(f"<div class='warn'>{_esc(w)}</div>")
    # results table
    parts.append("<h2>Results</h2><table><tr><th>quantity</th><th>pair</th><th>verdict</th><th>worst margin</th>"
                 "<th>at</th><th>measured</th><th>limit</th><th>points</th><th>failing</th></tr>")
    for r in ev.results:
        tag = "info" if r.passed is None else ("pass" if r.passed else "fail")
        word = "info" if r.passed is None else ("PASS" if r.passed else "FAIL")
        parts.append(f"<tr><td>{_esc(TITLES.get(r.quantity, r.quantity))}</td><td>{_esc(r.pair)}</td>"
                     f"<td><span class='tag {tag}'>{word}</span></td>"
                     f"<td class='num'>{'–' if r.worst_margin is None else f'{r.worst_margin:+.2f} {r.unit}'}</td>"
                     f"<td>{_esc(_xfmt(r))}</td><td class='num'>{_measured(r)}</td>"
                     f"<td class='num'>{_fmt(r.limit_at_worst)}</td><td class='num'>{r.n_points}</td><td class='num'>{r.n_fail}</td></tr>")
    parts.append("</table>")
    if embed_figures:
        parts.append("<h2>Margins</h2><div class='fig'>" + fig_to_svg(margin_bar(ev)) + "</div>")
        parts.append("<h2>Curves</h2>")
        for r in ev.results:
            if r.is_scalar:
                continue
            parts.append("<div class='fig'>" + fig_to_svg(plot_result(r)) + "</div>")
    parts.append(f"<h2>Limit provenance</h2><div class='small'>{_esc(ct.standard)} — {_esc(ct.clause)}<br>{_esc(ct.provenance)}</div>")
    parts.append("</body></html>")
    return "".join(parts)


def write_html(result: RunResult, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(render_html(result), encoding="utf-8")
    return path
