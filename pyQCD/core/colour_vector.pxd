from complex cimport Complex
from lattice_colour_vector cimport LatticeColourVector


cdef extern from "types.hpp":
    cdef cppclass ColourVector:
        ColourVector() except +
        ColourVector adjoint()
        Complex& operator[](int) except +
        LatticeColourVector broadcast() except +


    cdef ColourVector zeros "ColourVector::Zero"()
    cdef ColourVector ones "ColourVector::Ones"()
    cdef void mat_assign(ColourVector&, const int, const int, const Complex)
    cdef void mat_assign(ColourVector*, const int, const int, const Complex)