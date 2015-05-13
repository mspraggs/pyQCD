from scalar cimport Scalar

cdef extern from "types.hpp":
    cdef cppclass Complex:
        Complex() except +
        Complex(const Scalar, const Scalar) except +