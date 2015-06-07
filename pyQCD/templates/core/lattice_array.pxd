from libcpp.vector cimport vector

from {{ matrixdef.array_name|to_underscores }} cimport {{ matrixdef.array_name }}
from layout cimport Layout


cdef extern from "types.hpp":
    cdef cppclass {{ matrixdef.lattice_array_name }}:
        {{ matrixdef.lattice_array_name }}() except +
        {{ matrixdef.lattice_array_name }}(const Layout&, const {{ matrixdef.array_name }}) except +
        {{ matrixdef.array_name }}& operator[](const unsigned int)
        {{ matrixdef.array_name }}& operator()(const unsigned int)
        {{ matrixdef.array_name }}& operator()(const vector[unsigned int]&)
        unsigned int volume()
        unsigned int num_dims()
        const vector[unsigned int]& lattice_shape()