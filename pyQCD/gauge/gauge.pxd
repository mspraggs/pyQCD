from pyQCD.core.atomics cimport Real
from pyQCD.core cimport core


cdef extern from "gauge/gauge_action.hpp" namespace "pyQCD::gauge":
    cdef cppclass _GaugeAction "pyQCD::gauge::Action<pyQCD::Real, pyQCD::num_colours>":
        _GaugeAction(const Real) except +

cdef extern from "gauge/wilson_action.hpp" namespace "pyQCD::gauge":
    cdef cppclass _WilsonGaugeAction "pyQCD::gauge::WilsonAction<pyQCD::Real, pyQCD::num_colours>"(_GaugeAction):
        _WilsonGaugeAction(const Real, const core._Layout&) except +

cdef extern from "gauge/rectangle_action.hpp" namespace "pyQCD::gauge":
    cdef cppclass _RectangleGaugeAction "pyQCD::gauge::RectangleAction<pyQCD::Real, pyQCD::num_colours>"(_GaugeAction):
        _RectangleGaugeAction(const Real, const core._Layout&, const Real) except +

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
