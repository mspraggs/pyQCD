from complex cimport Complex


cdef extern from "types.hpp":
    cdef cppclass ColourVector:
        ColourVector() except +
        ColourVector(const ColourVector&) except +
        ColourVector adjoint()
        Complex& operator[](int) except +


    cdef ColourVector zeros "ColourVector::Zero"()
    cdef ColourVector ones "ColourVector::Ones"()
    cdef void mat_assign(ColourVector&, const int, const int, const Complex)
    cdef void mat_assign(ColourVector*, const int, const int, const Complex)