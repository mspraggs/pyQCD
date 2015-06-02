from complex cimport Complex
from lattice_colour_matrix cimport LatticeColourMatrix


cdef extern from "types.hpp":
    cdef cppclass ColourMatrix:
        ColourMatrix() except +
        ColourMatrix adjoint()
        Complex& operator()(int, int) except +
        LatticeColourMatrix broadcast() except +


    cdef ColourMatrix zeros "ColourMatrix::Zero"()
    cdef ColourMatrix ones "ColourMatrix::Ones"()
    cdef ColourMatrix identity "ColourMatrix::Identity"()
    cdef void mat_assign(ColourMatrix&, const int, const int, const Complex)
    cdef void mat_assign(ColourMatrix*, const int, const int, const Complex)