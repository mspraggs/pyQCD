from __future__ import absolute_import

import pytest

from pyQCD import algorithms, core, fermions, gauge


@pytest.fixture
def gauge_field():
    """Generate a cold-start gauge field with the supplied lattice shape"""
    shape = (8, 4, 4, 4)
    gauge_field = core.LatticeColourMatrix(shape, len(shape))
    gauge_field.as_numpy.fill(0.0)

    for i in range(gauge_field.as_numpy.shape[-1]):
        gauge_field.as_numpy[..., i, i] = 1.0

    return gauge_field


@pytest.fixture
def action():
    """Create an instance of the Wilson gauge action"""
    shape = (8, 4, 4, 4)
    return gauge.WilsonGaugeAction(5.5, shape)

def test_heatbath_update(action, gauge_field):
    """Test heatbath_update method"""
    algorithms.heatbath_update(gauge_field, action, 1)

def test_conjugate_gradient(gauge_field):
    """Test conjugate_gradient"""

    action = fermions.WilsonFermionAction(0.1, gauge_field)
    rhs = core.LatticeColourVector([8, 4, 4, 4], 4)
    rhs[0, 0, 0, 0, 0, 0] = 1.0
    results = algorithms.conjugate_gradient(action, rhs, 1000, 1e-10)

    assert len(results) == 3