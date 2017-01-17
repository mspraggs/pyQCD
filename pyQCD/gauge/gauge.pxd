from pyQCD.core.atomics cimport Real
from pyQCD.core.layout cimport Layout


cdef extern from "types.hpp" namespace "pyQCD::python":
    cdef cppclass _GaugeAction "pyQCD::python::GaugeAction":
        _GaugeAction(const Real, const Layout&) except +

    cdef cppclass _WilsonGaugeAction "pyQCD::python::WilsonGaugeAction"(CGaugeAction):
        _WilsonGaugeAction(const Real, const Layout&) except +

cdef class GaugeAction:
    cdef _GaugeAction* instance

cdef class WilsonGaugeAction(GaugeAction):
    pass