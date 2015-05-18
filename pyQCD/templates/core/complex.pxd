cdef extern from "types.hpp":
    cdef cppclass Complex:
        Complex() except +
        Complex(const {{ precision }}, const {{ precision }}) except +
        {{ precision }} real()
        {{ precision }} imag()