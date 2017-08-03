cdef extern from "utils/random.hpp" namespace "pyQCD":
    cdef cppclass _RandGenerator "pyQCD::RandGenerator":
        _RandGenerator()