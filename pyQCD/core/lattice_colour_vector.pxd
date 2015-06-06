from libcpp.vector cimport vector

from colour_vector cimport ColourVector
from layout cimport Layout


cdef extern from "types.hpp":
    cdef cppclass LatticeColourVector:
        LatticeColourVector() except +
        LatticeColourVector(const Layout&, const ColourVector) except +
        ColourVector& operator[](const unsigned int)
        unsigned int volume()
        unsigned int num_dims()
        const vector[unsigned int]& lattice_shape()