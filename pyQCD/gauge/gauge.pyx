from pyQCD.core.core cimport LatticeColourMatrix
from pyQCD.core.layout cimport Layout

from gauge cimport GaugeAction, WilsonGaugeAction, _average_plaquette


cdef class GaugeAction:
    pass

cdef class WilsonGaugeAction:

    def __cinit__(self, float beta, gauge_field):
        self.instance = new _WilsonGaugeAction(
            beta, (<LatticeColourMatrix?>gauge_field).lexico_layout[0])

def average_plaquette(LatticeColourMatrix gauge_field):
    return _average_plaquette(gauge_field.instance[0])

def average_rectangle(LatticeColourMatrix gauge_field):
    return _average_rectangle(gauge_field.instance[0])