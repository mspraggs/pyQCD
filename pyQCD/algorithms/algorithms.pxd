from pyQCD.core cimport lattice_colour_matrix
from pyQCD.gauge cimport gauge

cdef extern from "algorithms/heatbath.hpp" namespace "pyQCD":
    cdef void _heatbath_update "pyQCD::heatbath_update"(
        lattice_colour_matrix.LatticeColourMatrix&,
        const gauge._GaugeAction&, const unsigned int)