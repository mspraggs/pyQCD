from cpython cimport Py_buffer

cimport layout
from operators cimport *
cimport colour_matrix
cimport lattice_colour_matrix
cimport colour_vector
cimport lattice_colour_vector


cdef class ColourMatrix:
    cdef colour_matrix.ColourMatrix* instance
    cdef int view_count
    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]

    cdef inline ColourMatrix _add_ColourMatrix_ColourMatrix(ColourMatrix self, ColourMatrix other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline ColourMatrix _mul_ColourMatrix_ColourMatrix(ColourMatrix self, ColourMatrix other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourMatrix _mul_ColourMatrix_LatticeColourMatrix(ColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.lexico_layout.shape(), self.site_size)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline ColourVector _mul_ColourMatrix_ColourVector(ColourMatrix self, ColourVector other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourVector _mul_ColourMatrix_LatticeColourVector(ColourMatrix self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.lexico_layout.shape(), self.site_size)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline ColourMatrix _sub_ColourMatrix_ColourMatrix(ColourMatrix self, ColourMatrix other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

cdef class LatticeColourMatrix:
    cdef lattice_colour_matrix.LatticeColourMatrix* instance
    cdef layout.Layout* lexico_layout
    cdef int view_count
    cdef int site_size
    cdef Py_ssize_t buffer_shape[3]
    cdef Py_ssize_t buffer_strides[3]

    cdef inline LatticeColourMatrix _add_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.lexico_layout.shape(), self.site_size)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline LatticeColourMatrix _mul_LatticeColourMatrix_ColourMatrix(LatticeColourMatrix self, ColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(other.lexico_layout.shape(), self.site_size)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourMatrix _mul_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.lexico_layout.shape(), self.site_size)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourVector _mul_LatticeColourMatrix_ColourVector(LatticeColourMatrix self, ColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(other.lexico_layout.shape(), self.site_size)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourVector _mul_LatticeColourMatrix_LatticeColourVector(LatticeColourMatrix self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.lexico_layout.shape(), self.site_size)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourMatrix _sub_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.lexico_layout.shape(), self.site_size)
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

cdef class ColourVector:
    cdef colour_vector.ColourVector* instance
    cdef int view_count
    cdef Py_ssize_t buffer_shape[1]
    cdef Py_ssize_t buffer_strides[1]

    cdef inline ColourVector _add_ColourVector_ColourVector(ColourVector self, ColourVector other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline ColourVector _sub_ColourVector_ColourVector(ColourVector self, ColourVector other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

cdef class LatticeColourVector:
    cdef lattice_colour_vector.LatticeColourVector* instance
    cdef layout.Layout* lexico_layout
    cdef int view_count
    cdef int site_size
    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]

    cdef inline LatticeColourVector _add_LatticeColourVector_LatticeColourVector(LatticeColourVector self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.lexico_layout.shape(), self.site_size)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline LatticeColourVector _sub_LatticeColourVector_LatticeColourVector(LatticeColourVector self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.lexico_layout.shape(), self.site_size)
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

