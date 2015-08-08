from complex cimport Complex


cdef extern from "types.hpp":
    cdef cppclass ColourMatrix:
        ColourMatrix() except +
        ColourMatrix adjoint()
        Complex& operator()(int, int) except +


    cdef ColourMatrix zeros "ColourMatrix::Zero"()
    cdef ColourMatrix ones "ColourMatrix::Ones"()
    cdef ColourMatrix identity "ColourMatrix::Identity"()
    cdef void mat_assign(ColourMatrix&, const int, const int, const Complex)
    cdef void mat_assign(ColourMatrix*, const int, const int, const Complex)