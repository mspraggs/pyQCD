from libcpp.vector cimport vector

from colour_matrix_array cimport ColourMatrixArray
from layout cimport Layout


cdef extern from "types.hpp":
    cdef cppclass GaugeField:
        GaugeField() except +
        GaugeField(const Layout&, const ColourMatrixArray) except +
        unsigned int volume()
        unsigned int num_dims()
        const vector[unsigned int]& lattice_shape()