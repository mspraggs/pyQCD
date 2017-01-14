cdef extern from "types.hpp" namespace "python":
    ctypedef double Real

    cdef cppclass Complex:
        Complex() except +
        Complex(const Real, const Real) except +
        Real real()
        Real imag()