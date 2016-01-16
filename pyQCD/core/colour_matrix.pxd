from complex cimport Complex


cdef extern from "types.hpp" namespace "python":
    cdef cppclass ColourMatrix:
        ColourMatrix() except +
        ColourMatrix(const ColourMatrix&) except +
        ColourMatrix adjoint()
        Complex& operator()(int, int) except +


    cdef ColourMatrix zeros "python::ColourMatrix::Zero"()
    cdef ColourMatrix ones "python::ColourMatrix::Ones"()
    cdef ColourMatrix identity "python::ColourMatrix::Identity"()
    cdef void mat_assign(ColourMatrix&, const int, const int, const Complex)
    cdef void mat_assign(ColourMatrix*, const int, const int, const Complex)