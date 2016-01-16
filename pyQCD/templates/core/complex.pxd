cdef extern from "types.hpp" namespace "python":
    cdef cppclass Complex:
        Complex() except +
        Complex(const {{ precision }}, const {{ precision }}) except +
        {{ precision }} real()
        {{ precision }} imag()