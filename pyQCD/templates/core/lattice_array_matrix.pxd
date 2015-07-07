from libcpp.vector cimport vector

from {{ typedef.element_type.cmodule }} cimport {{ typedef.element_type.cname }}
from {{ typedef.cmodule }} cimport {{ typedef.cname }}
from layout cimport Layout


cdef extern from "types.hpp":
    cdef cppclass {{ typedef.cname }}:
        {{ typedef.cname }}() except +
        {{ typedef.cname }}(const Layout&, const {{ typedef.element_type.cname }}) except +
        {{ typedef.element_type.cname }}& operator[](const unsigned int)
        {{ typedef.element_type.cname }}& operator()(const unsigned int)
        {{ typedef.element_type.cname }}& operator()(const vector[unsigned int]&)
        unsigned int volume()
        unsigned int num_dims()
        const vector[unsigned int]& lattice_shape()