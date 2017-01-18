from __future__ import absolute_import

from pyQCD import gauge


class TestGaugeAction(object):

    def test_constructor(self):
        """Test constructor of GaugeAction object"""
        action = gauge.GaugeAction()


class TestWilsonGaugeAction(object):

    def test_constructor(self):
        """Test construction of WilsonGaugeAction"""
        action = gauge.WilsonGaugeAction(5.5, [8, 8, 8, 8])
