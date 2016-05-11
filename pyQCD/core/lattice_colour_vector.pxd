from libcpp.vector cimport vector

from colour_vector cimport ColourVector
from layout cimport Layout


cdef extern from "types.hpp" namespace "python":
    cdef cppclass LatticeColourVector:
        LatticeColourVector() except +
        LatticeColourVector(const Layout&, const ColourVector, unsigned int site_size) except +
        ColourVector& operator[](const unsigned int)
        unsigned int local_volume()
        unsigned int num_dims()
        const vector[unsigned int]& lattice_shape()