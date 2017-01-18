from __future__ import absolute_import

from pyQCD import algorithms, core, gauge


def create_gauge_field(shape):
    """Generate a cold-start gauge field with the supplied lattice shape"""
    gauge_field = core.LatticeColourMatrix(shape, len(shape))
    gauge_field.as_numpy.fill(0.0)

    for i in range(gauge_field.as_numpy.shape[-1]):
        gauge_field.as_numpy[..., i, i] = 1.0

    return gauge_field

class TestHeatbath(object):

    def test_constructor(self):
        """Test construction of Heatbath object"""

        # First we need an action
        # TODO: Put this in a fixture
        action = gauge.WilsonGaugeAction(5.5, [8, 4, 4, 4])
        heatbath = algorithms.Heatbath(action)

    def test_update(self):
        """Test Heatbath.update method"""

        action = gauge.WilsonGaugeAction(5.5, [8, 4, 4, 4])
        gauge_field = create_gauge_field([8, 4, 4, 4])

        heatbath = algorithms.Heatbath(action)
        heatbath.update(gauge_field, 1)