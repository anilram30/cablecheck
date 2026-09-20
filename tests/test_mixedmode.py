import numpy as np
import pytest

from cablecheck.mixedmode import PortMap, from_mixed_mode, to_mixed_mode
from cablecheck.synth import PairSpec, make_pair


def test_portmap_parse_and_lookup():
    pm = PortMap.parse("A+near,A-near,A+far,A-far")
    assert pm.pairs == ["A"] and pm.ports_of("A", "far") == (2, 3)
    assert pm.to_spec() == "A+near,A-near,A+far,A-far"
    with pytest.raises(ValueError):
        PortMap.parse("A+near,A+near,A+far,A-far")
    with pytest.raises(ValueError):
        PortMap.parse("A+near,A-near,A+far")  # missing far -


def test_balanced_line_has_no_mode_conversion(f_grid):
    n = make_pair(PairSpec(length_m=3.0, n_segments=5), f_grid)
    mm = to_mixed_mode(n, PortMap.single_pair())
    assert np.allclose(mm.z0, [100, 100, 25, 25])
    assert np.max(np.abs(mm.param(("c", "A", "near"), ("d", "A", "near")))) < 1e-9
    assert np.max(np.abs(mm.param(("c", "A", "far"), ("d", "A", "near")))) < 1e-9
    # differential transmission is near lossless at low frequency
    assert abs(mm.param(("d", "A", "far"), ("d", "A", "near"))[0]) > 0.98


def test_asymmetry_creates_mode_conversion(f_grid):
    n = make_pair(PairSpec(length_m=3.0, n_segments=5, asym_c=0.05), f_grid)
    mm = to_mixed_mode(n, PortMap.single_pair())
    assert np.max(np.abs(mm.param(("c", "A", "far"), ("d", "A", "near")))) > 1e-3


def test_inverse_transform(f_grid):
    n = make_pair(PairSpec(length_m=2.0, n_segments=4, asym_c=0.02), f_grid)
    pm = PortMap.single_pair()
    back = from_mixed_mode(to_mixed_mode(n, pm), pm)
    assert np.allclose(back.s, n.s, atol=1e-12)
    assert np.allclose(back.z0, 50.0)
