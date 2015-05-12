from layout cimport Layout

cdef extern from "types.hpp":
    cdef cppclass {{ matrixdef.lattice_matrix_name }}:
        {{ matrixdef.lattice_matrix_name }}() except +
        {{ matrixdef.lattice_matrix_name }}(const Layout&, const {{ matrixdef.matrix_name }}) except +