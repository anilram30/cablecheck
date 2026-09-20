"""
Single-ended -> mixed-mode conversion (Bockelman & Eisenstadt, 1995) for an
arbitrary number of balanced pairs, driven by a *port map* that says which
physical VNA port sits on which wire of which pair at which end of the cable.

For a pair whose + and - wires are on single-ended ports p and n (both with
reference impedance Z0) the differential- and common-mode power waves are

    a_d = (a_p - a_n)/sqrt(2),   a_c = (a_p + a_n)/sqrt(2)

and the same for b.  Stacking these rows for every (pair, end) into the
real orthogonal matrix M gives  S_mm = M S M^T  with reference impedances
Z_d = 2 Z0 for the differential ports and Z_c = Z0/2 for the common ports.
Ports that are not part of a pair are carried through unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .network import Network

__all__ = ["PortAssignment", "PortMap", "to_mixed_mode", "MixedModeNetwork"]

ENDS = ("near", "far")


@dataclass(frozen=True)
class PortAssignment:
    """Physical meaning of one single-ended VNA port."""

    pair: str | None      # e.g. "A"; None for an unpaired (single-ended) port
    end: str              # "near" or "far"
    polarity: str = "+"   # "+" or "-"

    def __post_init__(self):
        if self.end not in ENDS:
            raise ValueError(f"end must be one of {ENDS}")
        if self.polarity not in ("+", "-"):
            raise ValueError("polarity must be '+' or '-'")


class PortMap:
    """Ordered list of :class:`PortAssignment`, one per single-ended port."""

    def __init__(self, assignments: Iterable[PortAssignment]):
        self.assignments = list(assignments)
        self._check()

    # ----------------------------------------------------------- parsing
    @classmethod
    def parse(cls, spec: str) -> "PortMap":
        """Parse ``"A+near,A-near,A+far,A-far"`` (one token per port, in port
        order).  Token grammar: ``<pair><polarity><end>``; ``se:<end>`` for an
        unpaired port."""
        out = []
        for tok in [t.strip() for t in spec.split(",") if t.strip()]:
            if tok.startswith("se:"):
                out.append(PortAssignment(None, tok[3:], "+"))
                continue
            i = max(tok.rfind("+"), tok.rfind("-"))
            if i <= 0:
                raise ValueError(f"bad port token {tok!r}")
            out.append(PortAssignment(tok[:i], tok[i + 1:], tok[i]))
        return cls(out)

    @classmethod
    def single_pair(cls) -> "PortMap":
        """The standard 4-port hook-up: ports 1,2 = near end (+,-), 3,4 = far end (+,-)."""
        return cls.parse("A+near,A-near,A+far,A-far")

    @classmethod
    def two_pairs(cls) -> "PortMap":
        """8-port: A near (1,2), A far (3,4), B near (5,6), B far (7,8)."""
        return cls.parse("A+near,A-near,A+far,A-far,B+near,B-near,B+far,B-far")

    def _check(self):
        seen = {}
        for k, a in enumerate(self.assignments):
            if a.pair is None:
                continue
            key = (a.pair, a.end, a.polarity)
            if key in seen:
                raise ValueError(f"duplicate assignment {key} at ports {seen[key]} and {k}")
            seen[key] = k
        for (pair, end, pol) in list(seen):
            other = (pair, end, "-" if pol == "+" else "+")
            if other not in seen:
                raise ValueError(f"pair {pair} {end}: missing {other[2]} wire")

    def __len__(self):
        return len(self.assignments)

    @property
    def pairs(self) -> list[str]:
        out = []
        for a in self.assignments:
            if a.pair is not None and a.pair not in out:
                out.append(a.pair)
        return out

    def ports_of(self, pair: str, end: str) -> tuple[int, int]:
        """(index of + wire, index of - wire)."""
        p = n = None
        for k, a in enumerate(self.assignments):
            if a.pair == pair and a.end == end:
                if a.polarity == "+":
                    p = k
                else:
                    n = k
        if p is None or n is None:
            raise KeyError(f"pair {pair!r} end {end!r} not in port map")
        return p, n

    def groups(self) -> list[tuple[str, str]]:
        """Distinct (pair, end) combinations in port order."""
        out = []
        for a in self.assignments:
            if a.pair is not None and (a.pair, a.end) not in out:
                out.append((a.pair, a.end))
        return out

    def to_spec(self) -> str:
        return ",".join(f"se:{a.end}" if a.pair is None else f"{a.pair}{a.polarity}{a.end}"
                        for a in self.assignments)


class MixedModeNetwork(Network):
    """A :class:`Network` whose ports are modes, addressed by ``(mode, pair, end)``."""

    def __init__(self, f, s, z0, labels: list[tuple[str, str | None, str]], name=""):
        super().__init__(f, s, z0, name=name, port_names=[":".join(str(x) for x in l) for l in labels])
        self.labels = labels

    def index(self, mode: str, pair: str | None, end: str) -> int:
        try:
            return self.labels.index((mode, pair, end))
        except ValueError:
            raise KeyError(f"no mode port {(mode, pair, end)}") from None

    def param(self, out: tuple[str, str | None, str], inp: tuple[str, str | None, str]) -> np.ndarray:
        """Complex mixed-mode parameter S_{out,inp}, e.g.
        ``param(("d","A","far"), ("d","A","near"))`` is S_dd21 of pair A."""
        return self.s[:, self.index(*out), self.index(*inp)]


def to_mixed_mode(net: Network, pmap: PortMap) -> MixedModeNetwork:
    if len(pmap) != net.nports:
        raise ValueError(f"port map has {len(pmap)} entries but network has {net.nports} ports")
    n = net.nports
    rows = []
    labels: list[tuple[str, str | None, str]] = []
    z_mm = []
    # differential ports first, then common, then single-ended, in group order
    groups = pmap.groups()
    for mode in ("d", "c"):
        for (pair, end) in groups:
            p, q = pmap.ports_of(pair, end)
            if not np.isclose(net.z0[p], net.z0[q]):
                raise ValueError(f"pair {pair} {end}: + and - ports have different Z0; renormalise first")
            r = np.zeros(n)
            if mode == "d":
                r[p], r[q] = 1 / np.sqrt(2), -1 / np.sqrt(2)
                z_mm.append(2 * net.z0[p])
            else:
                r[p], r[q] = 1 / np.sqrt(2), 1 / np.sqrt(2)
                z_mm.append(net.z0[p] / 2)
            rows.append(r)
            labels.append((mode, pair, end))
    for k, a in enumerate(pmap.assignments):
        if a.pair is None:
            r = np.zeros(n)
            r[k] = 1.0
            rows.append(r)
            labels.append(("s", None, a.end))
            z_mm.append(net.z0[k])
    m = np.array(rows)
    s_mm = m[None] @ net.s @ m.T[None]
    return MixedModeNetwork(net.f, s_mm, np.array(z_mm), labels, name=net.name)


def from_mixed_mode(mm: MixedModeNetwork, pmap: PortMap) -> Network:
    """Inverse transform (used by the synthetic generator and for tests)."""
    n = len(pmap)
    rows = []
    z0 = np.zeros(n)
    for (mode, pair, end) in mm.labels:
        r = np.zeros(n)
        if mode == "s":
            k = [i for i, a in enumerate(pmap.assignments) if a.pair is None and a.end == end][0]
            r[k] = 1.0
            z0[k] = mm.z0[mm.labels.index((mode, pair, end))]
        else:
            p, q = pmap.ports_of(pair, end)
            if mode == "d":
                r[p], r[q] = 1 / np.sqrt(2), -1 / np.sqrt(2)
                z0[p] = z0[q] = mm.z0[mm.labels.index((mode, pair, end))] / 2
            else:
                r[p], r[q] = 1 / np.sqrt(2), 1 / np.sqrt(2)
        rows.append(r)
    m = np.array(rows)
    s = m.T[None] @ mm.s @ m[None]
    return Network(mm.f, s, z0, name=mm.name)
