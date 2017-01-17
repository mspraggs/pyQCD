from libcpp.vector cimport vector

from {{ typedef.element_type.cmodule }} cimport {{ typedef.element_type.cname }}
from layout cimport Layout


cdef extern from "types.hpp" namespace "pyQCD::python":
    cdef cppclass {{ typedef.cname }}:
        {{ typedef.cname }}() except +
        {{ typedef.cname }}(const Layout&, const {{ typedef.element_type.cname }}, unsigned int site_size) except +
        {{ typedef.element_type.cname }}& operator[](const unsigned int)
        unsigned int volume()
        unsigned int num_dims()
        const vector[unsigned int]& lattice_shape()