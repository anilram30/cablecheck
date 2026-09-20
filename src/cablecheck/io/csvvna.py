"""
Generic CSV export reader for VNA trace dumps (Keysight/Rohde & Schwarz
"save trace as CSV" style files).

The layout of these files varies between instruments, so the reader is
explicit rather than clever: the caller says which columns hold which
parameter.  Lines that do not start with a number are treated as header /
comment lines, which covers the ``BEGIN CH1_DATA`` / ``END`` wrappers, the
``Freq(Hz), S11(REAL), S11(IMAG), ...`` header row and quoted metadata.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..network import Network

_NUM = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*[,;\t ]")


def read_vna_csv(path: str | Path, nports: int, columns: dict[tuple[int, int], tuple[int, int]],
                 fmt: str = "RI", freq_col: int = 0, freq_unit: float = 1.0, z0: float = 50.0,
                 delimiter: str | None = None) -> Network:
    """Parse a CSV trace dump into a :class:`Network`.

    Parameters
    ----------
    columns
        ``{(i, j): (col_x, col_y)}`` mapping zero-based S-parameter index to the
        two zero-based column indices holding it (re/im, mag/deg or dB/deg
        depending on ``fmt``).  Parameters not listed are set to zero.
    fmt
        ``"RI"``, ``"MA"`` or ``"DB"``.
    freq_unit
        multiplier that converts the frequency column to Hz.
    """
    path = Path(path)
    rows: list[list[float]] = []
    comments: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        if _NUM.match(raw + ","):
            parts = re.split(r"[,;\t ]+" if delimiter is None else re.escape(delimiter), raw.strip())
            try:
                rows.append([float(p) for p in parts if p != ""])
            except ValueError:
                comments.append(raw.strip())
        else:
            comments.append(raw.strip())
    if not rows:
        raise ValueError(f"no numeric rows found in {path}")
    width = min(len(r) for r in rows)
    data = np.array([r[:width] for r in rows], dtype=float)
    f = data[:, freq_col] * freq_unit
    # keep only the first monotone sweep (files may contain several traces stacked)
    stop = np.flatnonzero(np.diff(f) <= 0)
    if stop.size:
        data = data[: stop[0] + 1]
        f = f[: stop[0] + 1]
    s = np.zeros((f.size, nports, nports), dtype=complex)
    fmt = fmt.upper()
    for (i, j), (cx, cy) in columns.items():
        x, y = data[:, cx], data[:, cy]
        if fmt == "RI":
            s[:, i, j] = x + 1j * y
        elif fmt == "MA":
            s[:, i, j] = x * np.exp(1j * np.radians(y))
        elif fmt == "DB":
            s[:, i, j] = 10 ** (x / 20) * np.exp(1j * np.radians(y))
        else:
            raise ValueError(f"unknown format {fmt}")
    return Network(f, s, np.full(nports, z0), name=path.name, comments=comments[:20])
