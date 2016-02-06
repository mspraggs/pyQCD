from libcpp.vector cimport vector

from colour_matrix cimport ColourMatrix
from layout cimport Layout


cdef extern from "types.hpp" namespace "python":
    cdef cppclass LatticeColourMatrix:
        LatticeColourMatrix() except +
        LatticeColourMatrix(const Layout&, const ColourMatrix, unsigned int site_size) except +
        ColourMatrix& operator[](const unsigned int)
        unsigned int volume()
        unsigned int num_dims()
        const vector[unsigned int]& lattice_shape()