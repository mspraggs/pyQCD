from {{ matrixdef.matrix_name|to_underscores }} cimport {{ matrixdef.matrix_name }}
from {{ matrixdef.lattice_array_name|to_underscores }} cimport {{ matrixdef.lattice_array_name }}


cdef extern from "types.hpp":
    cdef cppclass {{ matrixdef.array_name }}:
        {{ matrixdef.array_name }}() except +
        {{ matrixdef.array_name }}(int, {{ matrixdef.matrix_name }})
        {{ matrixdef.array_name }} adjoint()
        {{ matrixdef.matrix_name }}& operator[](const int) except +
        const {{ matrixdef.lattice_array_name }} broadcast()
        {{ matrixdef.lattice_array_name }} broadcast() except +