from atomics cimport Complex


cdef extern from "core/types.hpp" namespace "pyQCD::python":
    cdef cppclass ColourMatrix:
        ColourMatrix() except +
        ColourMatrix(const ColourMatrix&) except +
        ColourMatrix adjoint()
        Complex& operator()(int, int) except +


    cdef ColourMatrix zeros "pyQCD::python::ColourMatrix::Zero"()
    cdef ColourMatrix ones "pyQCD::python::ColourMatrix::Ones"()
    cdef ColourMatrix identity "pyQCD::python::ColourMatrix::Identity"()
    cdef void mat_assign(ColourMatrix&, const int, const int, const Complex)
    cdef void mat_assign(ColourMatrix*, const int, const int, const Complex)