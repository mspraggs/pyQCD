from pyQCD.core.core cimport LatticeColourMatrix
from pyQCD.core.layout cimport Layout

from gauge cimport GaugeAction, WilsonGaugeAction


cdef class GaugeAction:

    def __cinit__(self, float beta, gauge_field):
        raise NotImplementedError("Cannot create empty gauge action. Use a "
                                  "derived type.")

cdef class WilsonGaugeAction:

    def __cinit__(self, float beta, gauge_field):
        self.instance = new CWilsonGaugeAction(
            beta, (<LatticeColourMatrix?>gauge_field).lexico_layout[0])