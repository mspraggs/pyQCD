from libcpp.vector cimport vector

from {{ matrixdef.matrix_name|to_underscores }} cimport {{ matrixdef.matrix_name }}
from layout cimport Layout


cdef extern from "types.hpp":
    cdef cppclass {{ matrixdef.lattice_matrix_name }}:
        {{ matrixdef.lattice_matrix_name }}() except +
        {{ matrixdef.lattice_matrix_name }}(const Layout&, const {{ matrixdef.matrix_name }}) except +
        {{ matrixdef.matrix_name }}& operator[](const unsigned int)
        {{ matrixdef.matrix_name }}& operator()(const unsigned int)
        {{ matrixdef.matrix_name }}& operator()(const vector[unsigned int]&)
        unsigned int volume()
        unsigned int num_dims()
        const vector[unsigned int]& lattice_shape()