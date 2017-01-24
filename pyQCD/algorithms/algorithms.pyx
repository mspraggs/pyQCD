from pyQCD.core.core cimport LatticeColourMatrix
from pyQCD.gauge.gauge cimport GaugeAction

from algorithms cimport _heatbath_update


def heatbath_update(LatticeColourMatrix gauge_field,
                    GaugeAction action, int num_updates):
    _heatbath_update(gauge_field.instance[0], action.instance[0], num_updates)
