from colour_matrix cimport ColourMatrix
from gauge_field cimport GaugeField


cdef extern from "types.hpp":
    cdef cppclass ColourMatrixArray:
        ColourMatrixArray() except +
        ColourMatrixArray(int, ColourMatrix)
        ColourMatrixArray adjoint()
        ColourMatrix& operator[](const int) except +
        GaugeField broadcast() except +
        void resize(unsigned int) except +
        unsigned int size()