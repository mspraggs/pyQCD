cdef extern from "types.hpp":
    cdef cppclass {{ matrixdef.array_name }}:
        {{ matrixdef.array_name }}() except +
        {{ matrixdef.array_name }}(int, {{ matrixdef.matrix_name }})
        {{ matrixdef.array_name }} adjoint()
        const {{ matrixdef.matrix_name }} operator[](const int) except +