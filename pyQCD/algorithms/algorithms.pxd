from pyQCD.core cimport lattice_colour_matrix
from pyQCD.gauge cimport gauge

cdef extern from "types.hpp" namespace "pyQCD::python":
    cdef cppclass _Heatbath "pyQCD::python::Heatbath":
        _Heatbath(const gauge._GaugeAction&) except +
        void update(const lattice_colour_matrix.LatticeColourMatrix&, const unsigned int) except +

cdef class Heatbath:
    cdef _Heatbath* instance