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

def test_conjugate_gradient_unprec(gauge_field):
    """Test conjugate_gradient_unprec"""

    action = fermions.WilsonFermionAction(0.1, gauge_field, [0] * 4)
    rhs = core.LatticeColourVector(gauge_field.layout, 4)
    rhs[0, 0, 0, 0, 0, 0] = 1.0
    results = algorithms.conjugate_gradient_unprec(action, rhs, 1000, 1e-10)

    assert len(results) == 3

def test_conjugate_gradient_eoprec(gauge_field):
    """Test conjugate_gradient_eoprec"""

    eo_layout = core.EvenOddLayout(gauge_field.layout.shape)

    rhs = core.LatticeColourVector(gauge_field.layout, 4)
    rhs[0, 0, 0, 0, 0, 0] = 1.0

    gauge_field.change_layout(eo_layout)
    rhs.change_layout(eo_layout)

    action = fermions.WilsonFermionAction(0.1, gauge_field, [0] * 4)

    results = algorithms.conjugate_gradient_eoprec(action, rhs, 1000, 1e-10)

    assert len(results) == 3
    assert results[1] < 1000
    assert results[2] < 1e-10