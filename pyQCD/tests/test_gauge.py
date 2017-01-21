from __future__ import absolute_import

import numpy as np
import pytest

from pyQCD import gauge


class TestGaugeAction(object):

    def test_constructor(self):
        """Test constructor of GaugeAction object"""
        with pytest.raises(NotImplementedError):
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


def test_hot_start():
    """Test construction of hot start gauge field"""

    shape = (8, 4, 4, 4)
    gauge_field = gauge.hot_start(shape)

    num_colours = gauge_field.as_numpy.shape[-1]
    id = np.identity(num_colours)

    array_view = gauge_field.as_numpy

    for index in np.ndindex(*(shape + (num_colours,))):
        link = array_view[index]
        assert np.allclose(np.linalg.det(link), 1.0)
        assert np.allclose(np.dot(link, np.conj(link.T)), id)

