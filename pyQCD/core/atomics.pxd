cdef extern from "globals.hpp" namespace "pyQCD":
    ctypedef double Real

    cdef cppclass Complex:
        Complex() except +
        Complex(const Real, const Real) except +
        Real real()
        Real imag()