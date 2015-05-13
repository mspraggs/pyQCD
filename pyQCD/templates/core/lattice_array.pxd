from {{ matrixdef.array_name|to_underscores }} cimport {{ matrixdef.array_name }}
from layout cimport Layout


cdef extern from "types.hpp":
    cdef cppclass {{ matrixdef.lattice_array_name }}:
        {{ matrixdef.lattice_array_name }}() except +
        {{ matrixdef.lattice_array_name }}(const Layout&, const {{ matrixdef.array_name }}) except +