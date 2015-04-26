from eigen cimport ColourMatrix

cdef extern from "gauge_field.hpp" namespace "pyQCD":
    cdef cppclass GaugeField "GaugeField<3, double>":
        GaugeField()
        GaugeField(int, ColourMatrix&)
        GaugeField adjoint()