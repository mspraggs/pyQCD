cdef extern from "types.hpp" namespace "python":
    cdef cppclass Complex:
        Complex() except +
        Complex(const double, const double) except +
        double real()
        double imag()