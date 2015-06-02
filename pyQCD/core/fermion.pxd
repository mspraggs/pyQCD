from colour_vector cimport ColourVector
from fermion_field cimport FermionField


cdef extern from "types.hpp":
    cdef cppclass Fermion:
        Fermion() except +
        Fermion(int, ColourVector)
        Fermion adjoint()
        ColourVector& operator[](const int) except +
        const FermionField broadcast()
        FermionField broadcast() except +
        void resize(unsigned int) except +
        unsigned int size()