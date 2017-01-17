from pyQCD.core.core cimport LatticeColourMatrix
from pyQCD.gauge.gauge cimport GaugeAction

from algorithms cimport Heatbath


cdef class Heatbath:

    def __cinit__(self, GaugeAction action):
        self.instance = new _Heatbath(action.instance[0])

    def update(self, LatticeColourMatrix gauge_field, int num_updates):
        self.instance.update(gauge_field.instance[0], num_updates)