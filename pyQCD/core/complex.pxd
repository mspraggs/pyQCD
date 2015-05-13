from real cimport Real

cdef extern from "types.hpp":
    cdef cppclass Complex:
        Complex() except +
        Complex(const Real, const Real) except +