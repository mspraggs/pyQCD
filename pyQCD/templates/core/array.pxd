from {{ typedef.cmodule }} cimport {{ typedef.cname }}
{# from {{ matrixdef.lattice_array_name|to_underscores }} cimport {{ matrixdef.lattice_array_name }} #}


cdef extern from "types.hpp":
    cdef cppclass {{ typedef.cname }}:
        {{ typedef.cname }}() except +
        {{ typedef.cname }}(int, {{ typedef.element_type.cname }})
        {{ typedef.cname }} adjoint()
        {{ typedef.element_type.cname }}& operator[](const int) except +
        {#const {{ matrixdef.lattice_array_name }} broadcast()#}
        {#{{ matrixdef.lattice_array_name }} broadcast() except +#}
        void resize(unsigned int) except +
        unsigned int size()