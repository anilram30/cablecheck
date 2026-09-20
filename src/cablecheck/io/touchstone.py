"""
Touchstone reader/writer (v1.x and the common subset of v2.0).

Supported on read
-----------------
* extension ``.sNp`` (any N) or ``.ts``; the port count is taken from the
  ``[Number of Ports]`` keyword (v2) or the extension (v1)
* option line ``# <unit> S <MA|DB|RI> R <z0>`` in any order / case; only
  S-parameters are accepted (Y/Z/G/H files are rejected explicitly)
* v1 2-port ordering quirk (S11 S21 S12 S22) is handled
* v2 keywords ``[Version] [Number of Ports] [Reference] [Matrix Format]
  (Full|Lower|Upper) [Number of Frequencies] [Network Data] [End]``;
  ``[Two-Port Data Order]`` 12_21 / 21_12
* ``!`` comments anywhere; noise-parameter block of 2-port v1 files ignored

Written files are v1 with ``# Hz S RI R <z0>`` and 15 significant digits so
that a write/read round trip is loss-free.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..network import Network

_UNIT = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}


class TouchstoneError(ValueError):
    pass


def _ports_from_name(path: Path) -> int | None:
    m = re.fullmatch(r"\.s(\d+)p", path.suffix.lower())
    return int(m.group(1)) if m else None


def read_touchstone(path: str | Path) -> Network:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_touchstone(text, nports_hint=_ports_from_name(path), name=path.name)


def parse_touchstone(text: str, nports_hint: int | None = None, name: str = "") -> Network:
    unit, fmt, z0_opt = 1e9, "MA", 50.0  # Touchstone defaults
    comments: list[str] = []
    nports = nports_hint
    version = 1.0
    reference: np.ndarray | None = None
    matrix_format = "FULL"
    two_port_order = "21_12"  # v1 default: S11 S21 S12 S22
    numbers: list[float] = []
    saw_option = False
    in_data = version < 2  # v1: data starts after option line

    for raw in text.splitlines():
        line = raw.split("!", 1)
        if len(line) == 2 and line[1].strip():
            comments.append(line[1].strip())
        body = line[0].strip()
        if not body:
            continue
        if body.startswith("["):
            key, _, val = body.partition("]")
            key = key[1:].strip().upper()
            val = val.strip()
            if key == "VERSION":
                version = float(val)
                in_data = False
            elif key == "NUMBER OF PORTS":
                nports = int(val)
            elif key == "REFERENCE":
                reference = np.array([float(x) for x in val.split()]) if val else np.array([])
            elif key == "MATRIX FORMAT":
                matrix_format = val.upper()
            elif key == "TWO-PORT DATA ORDER":
                two_port_order = val.upper()
            elif key == "NETWORK DATA":
                in_data = True
            elif key == "END":
                break
            elif key in ("NOISE DATA",):
                in_data = False
            # other keywords ([Number of Frequencies], [Begin Information], ...) ignored
            continue
        if body.startswith("#"):
            if saw_option:
                continue  # some writers repeat it; ignore
            saw_option = True
            toks = body[1:].upper().split()
            i = 0
            while i < len(toks):
                t = toks[i]
                if t in _UNIT:
                    unit = _UNIT[t]
                elif t in ("MA", "DB", "RI"):
                    fmt = t
                elif t == "S":
                    pass
                elif t in ("Y", "Z", "G", "H"):
                    raise TouchstoneError(f"only S-parameter Touchstone files are supported (got {t})")
                elif t == "R":
                    i += 1
                    z0_opt = float(toks[i])
                i += 1
            if version < 2:
                in_data = True
            continue
        if not in_data:
            continue
        try:
            numbers.extend(float(x) for x in body.split())
        except ValueError as e:
            raise TouchstoneError(f"cannot parse data line {body!r}") from e

    if nports is None:
        raise TouchstoneError("cannot determine port count (use .sNp extension or [Number of Ports])")
    if not saw_option:
        raise TouchstoneError("missing option line starting with '#'")
    if reference is not None and reference.size == 0:
        reference = None
    if reference is not None and reference.size != nports:
        raise TouchstoneError("[Reference] entry count differs from port count")

    n = nports
    if matrix_format == "FULL":
        per_freq = 1 + 2 * n * n
    elif matrix_format in ("LOWER", "UPPER"):
        per_freq = 1 + n * (n + 1)
    else:
        raise TouchstoneError(f"unknown [Matrix Format] {matrix_format}")

    vals = np.asarray(numbers, dtype=float)
    # v1 2-port files may carry a trailing noise block: keep the leading
    # monotone-frequency part only.
    if n == 2 and version < 2 and vals.size % per_freq:
        nrec = vals.size // per_freq
        vals = vals[: nrec * per_freq]
    if vals.size % per_freq:
        raise TouchstoneError(f"data length {vals.size} is not a multiple of {per_freq} for {n} ports")
    rec = vals.reshape(-1, per_freq)
    f = rec[:, 0] * unit
    if n == 2 and version < 2:
        # trailing noise data: frequency restarts
        nz = np.flatnonzero(np.diff(f) <= 0)
        if nz.size:
            rec = rec[: nz[0] + 1]
            f = f[: nz[0] + 1]
    data = rec[:, 1:]
    x, y = data[:, 0::2], data[:, 1::2]
    if fmt == "MA":
        c = x * np.exp(1j * np.radians(y))
    elif fmt == "DB":
        c = 10 ** (x / 20.0) * np.exp(1j * np.radians(y))
    else:
        c = x + 1j * y

    if matrix_format == "FULL":
        s = c.reshape(-1, n, n)  # row-major: S11 S12 ... S1n S21 ...
        if n == 2 and two_port_order == "21_12":
            s = np.transpose(s, (0, 2, 1))  # file order is S11 S21 S12 S22
    else:
        s = np.zeros((c.shape[0], n, n), dtype=complex)
        k = 0
        for i in range(n):
            for j in range(i + 1):
                if matrix_format == "LOWER":
                    s[:, i, j] = c[:, k]
                    s[:, j, i] = c[:, k]
                else:  # UPPER: rows i, columns i..n-1
                    pass
                k += 1
        if matrix_format == "UPPER":
            k = 0
            s[:] = 0
            for i in range(n):
                for j in range(i, n):
                    s[:, i, j] = c[:, k]
                    s[:, j, i] = c[:, k]
                    k += 1
    z0 = reference if reference is not None else np.full(n, z0_opt)
    if np.any(np.diff(f) <= 0):
        raise TouchstoneError("frequencies are not strictly increasing")
    return Network(f, s, z0, name=name, comments=comments)


def write_touchstone(net: Network, path: str | Path, fmt: str = "RI",
                     unit: str = "Hz", comments: list[str] | None = None) -> Path:
    path = Path(path)
    n = net.nports
    if not np.allclose(net.z0, net.z0[0]):
        raise TouchstoneError("Touchstone v1 supports a single reference impedance; renormalise first")
    fmt = fmt.upper()
    unit = unit.upper()
    scale = _UNIT[unit]
    lines = []
    for c in (comments or []) + list(net.comments):
        lines.append("! " + c)
    lines.append(f"# {unit if unit != 'HZ' else 'Hz'} S {fmt} R {net.z0[0]:g}")
    per_line = 4 if n >= 3 else n * n
    for k in range(net.nfreq):
        s = net.s[k]
        if n == 2:
            order = [(0, 0), (1, 0), (0, 1), (1, 1)]
        else:
            order = [(i, j) for i in range(n) for j in range(n)]
        vals = []
        for (i, j) in order:
            v = s[i, j]
            if fmt == "RI":
                vals.append(f"{v.real:.15e} {v.imag:.15e}")
            elif fmt == "MA":
                vals.append(f"{abs(v):.15e} {np.degrees(np.angle(v)):.12f}")
            elif fmt == "DB":
                vals.append(f"{20 * np.log10(max(abs(v), 1e-300)):.12f} {np.degrees(np.angle(v)):.12f}")
            else:
                raise TouchstoneError(f"unknown format {fmt}")
        if n <= 2:
            lines.append(f"{net.f[k] / scale:.10f} " + " ".join(vals))
        else:
            first = True
            for i in range(n):
                row = vals[i * n:(i + 1) * n]
                for start in range(0, n, per_line):
                    chunk = " ".join(row[start:start + per_line])
                    if first:
                        lines.append(f"{net.f[k] / scale:.10f} {chunk}")
                        first = False
                    else:
                        lines.append(" " * 12 + chunk)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
