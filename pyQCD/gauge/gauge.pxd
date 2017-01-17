from pyQCD.core.atomics cimport Real
from pyQCD.core.layout cimport Layout


cdef extern from "types.hpp" namespace "pyQCD::python":
    cdef cppclass CGaugeAction "pyQCD::python::GaugeAction":
        CGaugeAction(const Real, const Layout&) except +

    cdef cppclass CWilsonGaugeAction "pyQCD::python::WilsonGaugeAction"(CGaugeAction):
        CWilsonGaugeAction(const Real, const Layout&) except +

cdef class GaugeAction:
    cdef CGaugeAction* instance

cdef class WilsonGaugeAction(GaugeAction):
    pass