from libcpp.vector cimport vector

from fermion cimport Fermion
from layout cimport Layout


cdef extern from "types.hpp":
    cdef cppclass FermionField:
        FermionField() except +
        FermionField(const Layout&, const Fermion) except +
        Fermion& operator[](const unsigned int)
        Fermion& operator()(const unsigned int)
        Fermion& operator()(const vector[unsigned int]&)
        unsigned int volume()
        unsigned int num_dims()
        const vector[unsigned int]& lattice_shape()