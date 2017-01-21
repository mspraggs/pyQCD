from __future__ import absolute_import

import numpy as np

from pyQCD import gauge


class TestGaugeAction(object):

    def test_constructor(self):
        """Test constructor of GaugeAction object"""
        action = gauge.GaugeAction()


class TestWilsonGaugeAction(object):

    def test_constructor(self):
        """Test construction of WilsonGaugeAction"""
        action = gauge.WilsonGaugeAction(5.5, [8, 8, 8, 8])


def test_cold_start():
    """Test construction of cold start gauge field"""

    shape = (8, 4, 4, 4)
    gauge_field = gauge.cold_start(shape)

    num_colours = gauge_field.as_numpy.shape[-1]
    id = np.identity(num_colours)

    array_view = gauge_field.as_numpy

    for index in np.ndindex(*(shape + (num_colours,))):
        assert np.allclose(array_view[index], id)
