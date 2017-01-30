from pyQCD.core.atomics cimport Real
from pyQCD.core cimport core


cdef extern from "gauge/types.hpp" namespace "pyQCD::python":
    cdef cppclass _GaugeAction "pyQCD::python::GaugeAction":
        _GaugeAction(const Real, const core.Layout&) except +

    cdef cppclass _WilsonGaugeAction "pyQCD::python::WilsonGaugeAction"(_GaugeAction):
        _WilsonGaugeAction(const Real, const core.Layout&) except +

    cdef cppclass _RectangleGaugeAction "pyQCD::python::RectangleGaugeAction"(_GaugeAction):
        _RectangleGaugeAction(const Real, const core.Layout&, const Real) except +

cdef extern from "gauge/plaquette.hpp" namespace "pyQCD::gauge":
    cdef Real _average_plaquette "pyQCD::gauge::average_plaquette"(const core._LatticeColourMatrix&) except +
    
cdef extern from "gauge/rectangle.hpp" namespace "pyQCD::gauge":
    cdef Real _average_rectangle "pyQCD::gauge::average_rectangle"(const core._LatticeColourMatrix&) except +

cdef class GaugeAction:
    cdef _GaugeAction* instance

cdef class WilsonGaugeAction(GaugeAction):
    pass

cdef class SymanzikGaugeAction(GaugeAction):
    pass

cdef class IwasakiGaugeAction(GaugeAction):
    pass
