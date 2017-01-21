from __future__ import absolute_import

import pytest

from pyQCD import algorithms, core, gauge


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


class TestHeatbath(object):

    def test_constructor(self, action):
        """Test construction of Heatbath object"""
        heatbath = algorithms.Heatbath(action)

    def test_update(self, action, gauge_field):
        """Test Heatbath.update method"""

        heatbath = algorithms.Heatbath(action)
        heatbath.update(gauge_field, 1)