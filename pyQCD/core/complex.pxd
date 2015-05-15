cdef extern from "types.hpp":
    cdef cppclass Complex:
        Complex() except +
        Complex(const double, const double) except +