from {{ typedef.element_type.cmodule }} cimport {{ typedef.element_type.cname }}
from {{ bcast_typedef.cmodule }} cimport {{ bcast_typedef.cname }}


cdef extern from "types.hpp":
    cdef cppclass {{ typedef.cname }}:
        {{ typedef.cname }}() except +
        {{ typedef.cname }}(int, {{ typedef.element_type.cname }})
        {{ typedef.cname }} adjoint()
        {{ typedef.element_type.cname }}& operator[](const int) except +
        {{ bcast_typedef.cname }} broadcast() except +
        void resize(unsigned int) except +
        unsigned int size()