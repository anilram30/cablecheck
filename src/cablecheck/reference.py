"""
Synthetic reference measurement set.

Five samples whose physics is known exactly, with a fixture, written as
Touchstone files plus a manifest.  The test-suite evaluates them and
compares every reported number against ``expected.json`` (frozen once,
regenerated deliberately with ``--refresh``) so that a code change that moves
a verdict or a margin is caught.

  good_15m        clean 15 m pair, passes 1000BASE-T1 link-segment limits
  lossy_15m       thin conductor + lossy dielectric: insertion loss fails above ~350 MHz
  ripple_15m      periodic 2.5 % capacitance ripple (capstan signature) + local defect:
                  return loss fails, impedance window fails
  unbalanced_10m  4 % capacitance asymmetry: LCL/LCTL fail
  twopair_5m      two coupled pairs (8-port): NEXT/FEXT exercised
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import write_touchstone
from .synth import (
    FixtureSpec,
    PairSpec,
    make_fixture,
    make_pair,
    make_two_pairs,
    synthesize_measurement,
)

SAMPLES = {
    "good_15m": dict(spec=PairSpec(length_m=15.0, d_wire_m=0.55e-3, tan_delta=0.0012, roughness=0.004, seed=11),
                     cable_type="1000base-t1-link-segment", length=15.0, ports=4),
    "lossy_15m": dict(spec=PairSpec(length_m=15.0, d_wire_m=0.28e-3, tan_delta=0.006, proximity=1.35, roughness=0.004, seed=12),
                      cable_type="1000base-t1-link-segment", length=15.0, ports=4),
    "ripple_15m": dict(spec=PairSpec(length_m=15.0, d_wire_m=0.55e-3, tan_delta=0.0012, ripple_amp=0.025, ripple_period_m=0.26,
                                     defect_pos_m=6.0, defect_amp=0.25, defect_width_m=0.08, roughness=0.004, n_segments=240, seed=13),
                       cable_type="1000base-t1-link-segment", length=15.0, ports=4),
    "unbalanced_10m": dict(spec=PairSpec(length_m=10.0, d_wire_m=0.5e-3, tan_delta=0.0015, asym_c=0.04, asym_r=0.1, roughness=0.004, seed=14),
                           cable_type="1000base-t1-link-segment", length=10.0, ports=4),
    "twopair_5m": dict(spec=PairSpec(length_m=5.0, d_wire_m=0.5e-3, tan_delta=0.0015, roughness=0.004, n_segments=25, seed=15),
                       cable_type="1000base-t1-link-segment", length=5.0, ports=8),
}

PORT_MAPS = {4: "A+near,A-near,A+far,A-far", 8: "A+near,A-near,A+far,A-far,B+near,B-near,B+far,B-far"}


def generate_reference_set(out: Path, n_points: int = 1200, fmin: float = 1e6, fmax: float = 600e6) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = np.linspace(fmin, fmax, n_points)
    fx = make_fixture(FixtureSpec(), f, name="fixture")
    paths = [write_touchstone(fx, out / "fixture_launch.s4p", comments=["synthetic launch fixture: ports 1,2 VNA side; 3,4 DUT side"])]
    # 2x-thru of the same fixture, for the bisection path
    from .deembed import embed
    from .network import Network
    ideal = Network(f, np.tile(np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], complex)[None], (f.size, 1, 1)),
                    np.full(4, 50.0))
    paths.append(write_touchstone(synthesize_measurement(embed(ideal, fx, fx), None, seed=99), out / "fixture_2xthru.s4p",
                                  comments=["synthetic 2x-thru of fixture_launch"]))
    manifest = {"frequency": {"fmin_hz": fmin, "fmax_hz": fmax, "points": n_points}, "fixture": "fixture_launch.s4p",
                "thru": "fixture_2xthru.s4p", "samples": {}}
    for name, cfg in SAMPLES.items():
        spec = cfg["spec"]
        if cfg["ports"] == 4:
            dut = make_pair(spec, f, name=name)
            fixture = fx
        else:
            spec_b = PairSpec(**{**spec.__dict__, "seed": spec.seed + 100})
            dut = make_two_pairs(spec, spec_b, f, coupling_l=0.02, coupling_c=0.015, imbalance=0.15, name=name)
            # 8-port fixture: block diagonal of two 4-port fixtures, reordered to (VNA A, VNA B | DUT A, DUT B)
            s8 = np.zeros((f.size, 8, 8), dtype=complex)
            s8[:, :4, :4] = fx.s
            s8[:, 4:, 4:] = fx.s
            order = [0, 1, 4, 5, 2, 3, 6, 7]
            fixture = Network(f, s8[:, order][:, :, order], np.full(8, 50.0), name="fixture8")
        meas = synthesize_measurement(dut, fixture, seed=hash(name) % 1000, name=name)
        ext = f".s{cfg['ports']}p"
        p = write_touchstone(meas, out / f"{name}{ext}", comments=[f"synthetic reference sample {name}",
                                                                 "measured through fixture_launch at both ends"])
        paths.append(p)
        # also the bare DUT (truth) for the de-embedding test
        write_touchstone(dut, out / f"{name}_truth{ext}", comments=["bare DUT without fixture (ground truth)"])
        manifest["samples"][name] = {"file": p.name, "truth": f"{name}_truth{ext}", "cable_type": cfg["cable_type"],
                                     "length_m": cfg["length"], "port_map": PORT_MAPS[cfg["ports"]],
                                     "spec": {k: v for k, v in spec.__dict__.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    paths.append(out / "manifest.json")
    return paths
