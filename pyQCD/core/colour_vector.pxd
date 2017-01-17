from atomics cimport Complex


cdef extern from "types.hpp" namespace "pyQCD::python":
    cdef cppclass ColourVector:
        ColourVector() except +
        ColourVector(const ColourVector&) except +
        ColourVector adjoint()
        Complex& operator[](int) except +


    cdef ColourVector zeros "pyQCD::python::ColourVector::Zero"()
    cdef ColourVector ones "pyQCD::python::ColourVector::Ones"()
    cdef void mat_assign(ColourVector&, const int, const int, const Complex)
    cdef void mat_assign(ColourVector*, const int, const int, const Complex)