from colour_matrix cimport ColourMatrix
from layout cimport Layout


cdef extern from "types.hpp":
    cdef cppclass LatticeColourMatrix:
        LatticeColourMatrix() except +
        LatticeColourMatrix(const Layout&, const ColourMatrix) except +
        unsigned int volume()
        unsigned int num_dims()