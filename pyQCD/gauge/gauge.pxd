from pyQCD.core.complex cimport Real
from pyQCD.core.layout cimport Layout


cdef extern from "types.hpp" namespace "python":
    cdef cppclass CGaugeAction "python::GaugeAction":
        CGaugeAction(const Real, const Layout&) except +

    cdef cppclass CWilsonGaugeAction "python::WilsonGaugeAction"(CGaugeAction):
        CWilsonGaugeAction(const Real, const Layout&) except +

cdef class GaugeAction:
    cdef CGaugeAction* instance

cdef class WilsonGaugeAction(GaugeAction):
    pass