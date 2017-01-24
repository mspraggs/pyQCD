from pyQCD.core cimport core
from pyQCD.gauge cimport gauge

cdef extern from "algorithms/heatbath.hpp" namespace "pyQCD":
    cdef void _heatbath_update "pyQCD::heatbath_update"(
        core._LatticeColourMatrix&,
        const gauge._GaugeAction&, const unsigned int)