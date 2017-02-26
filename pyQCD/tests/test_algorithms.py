from __future__ import absolute_import

import pytest

from pyQCD import algorithms, core, fermions, gauge


@pytest.fixture
def layout():
    return core.LexicoLayout((8, 4, 4, 4))

@pytest.fixture
def gauge_field(layout):
    """Generate a cold-start gauge field with the supplied lattice shape"""
    gauge_field = core.LatticeColourMatrix(layout, layout.ndims)
    gauge_field.as_numpy.fill(0.0)

    for i in range(gauge_field.as_numpy.shape[-1]):
        gauge_field.as_numpy[..., i, i] = 1.0

    return gauge_field


@pytest.fixture
def action(layout):
    """Create an instance of the Wilson gauge action"""
    return gauge.WilsonGaugeAction(5.5, layout)

def test_heatbath_update(action, gauge_field):
    """Test heatbath_update method"""
    algorithms.heatbath_update(gauge_field, action, 1)

def test_conjugate_gradient(gauge_field):
    """Test conjugate_gradient"""

    action = fermions.WilsonFermionAction(0.1, gauge_field, [0] * 4)
    rhs = core.LatticeColourVector(gauge_field.layout, 4)
    rhs[0, 0, 0, 0, 0, 0] = 1.0
    results = algorithms.conjugate_gradient(action, rhs, 1000, 1e-10)

    assert len(results) == 3