from cpython cimport Py_buffer
from libcpp.vector cimport vector

import numpy as np

cimport complex
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

    def __cinit__(self, ):
        self.view_count = 0
        self.instance = new colour_matrix.ColourMatrix(colour_matrix.zeros())

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = 3
        self.buffer_strides[0] = itemsize
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize * 3

        buffer.buf = <char*>self.instance
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 9
        buffer.ndim = 2

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer* buffer):
        pass

    property as_numpy:
        """Return a view to this object as a numpy array"""
        def __get__(self):
            out = np.asarray(self)
            out.dtype = complex
            return out

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out[:] = value

    def __add__(self, other):
        if type(self) is ColourMatrix and type(other) is ColourMatrix:
            return (<ColourMatrix>self)._add_ColourMatrix_ColourMatrix(<ColourMatrix>other)
        raise TypeError("Unsupported operand types for ColourMatrix.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrix _add_ColourMatrix_ColourMatrix(ColourMatrix self, ColourMatrix other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is ColourMatrix and type(other) is ColourMatrix:
            return (<ColourMatrix>self)._mul_ColourMatrix_ColourMatrix(<ColourMatrix>other)
        if type(self) is ColourMatrix and type(other) is LatticeColourMatrix:
            return (<ColourMatrix>self)._mul_ColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        if type(self) is ColourMatrix and type(other) is ColourVector:
            return (<ColourMatrix>self)._mul_ColourMatrix_ColourVector(<ColourVector>other)
        if type(self) is ColourMatrix and type(other) is LatticeColourVector:
            return (<ColourMatrix>self)._mul_ColourMatrix_LatticeColourVector(<LatticeColourVector>other)
        raise TypeError("Unsupported operand types for ColourMatrix.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrix _mul_ColourMatrix_ColourMatrix(ColourMatrix self, ColourMatrix other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourMatrix _mul_ColourMatrix_LatticeColourMatrix(ColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.lexico_layout.shape())
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline ColourVector _mul_ColourMatrix_ColourVector(ColourMatrix self, ColourVector other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourVector _mul_ColourMatrix_LatticeColourVector(ColourMatrix self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.lexico_layout.shape())
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    def __sub__(self, other):
        if type(self) is ColourMatrix and type(other) is ColourMatrix:
            return (<ColourMatrix>self)._sub_ColourMatrix_ColourMatrix(<ColourMatrix>other)
        raise TypeError("Unsupported operand types for ColourMatrix.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrix _sub_ColourMatrix_ColourMatrix(ColourMatrix self, ColourMatrix other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        raise TypeError("Unsupported operand types for ColourMatrix.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        raise TypeError("Unsupported operand types for ColourMatrix.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

cdef class LatticeColourMatrix:
    cdef lattice_colour_matrix.LatticeColourMatrix* instance
    cdef layout.Layout* lexico_layout
    cdef int view_count
    cdef Py_ssize_t buffer_shape[3]
    cdef Py_ssize_t buffer_strides[3]

    def __cinit__(self, shape):
        self.lexico_layout = new layout.LexicoLayout(shape)
        self.view_count = 0
        self.instance = new lattice_colour_matrix.LatticeColourMatrix(self.lexico_layout[0], colour_matrix.ColourMatrix(colour_matrix.zeros()))

    def __dealloc__(self):
        del self.instance
        del self.lexico_layout

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance[0].volume()
        self.buffer_strides[0] = itemsize * 9
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize
        self.buffer_shape[2] = 3
        self.buffer_strides[2] = itemsize * 3

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 9 * self.instance[0].volume()
        buffer.ndim = 3

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1
    property as_numpy:
        """Return a view to this object as a numpy array"""
        def __get__(self):
            out = np.asarray(self)
            out.dtype = complex
            return out

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out[:] = value

    def __add__(self, other):
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>self)._add_LatticeColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourMatrix _add_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.lexico_layout.shape())
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is LatticeColourMatrix and type(other) is ColourMatrix:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_ColourMatrix(<ColourMatrix>other)
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        if type(self) is LatticeColourMatrix and type(other) is ColourVector:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_ColourVector(<ColourVector>other)
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourVector:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_LatticeColourVector(<LatticeColourVector>other)
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourMatrix _mul_LatticeColourMatrix_ColourMatrix(LatticeColourMatrix self, ColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(other.lexico_layout.shape())
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourMatrix _mul_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.lexico_layout.shape())
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourVector _mul_LatticeColourMatrix_ColourVector(LatticeColourMatrix self, ColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(other.lexico_layout.shape())
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourVector _mul_LatticeColourMatrix_LatticeColourVector(LatticeColourMatrix self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.lexico_layout.shape())
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    def __sub__(self, other):
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>self)._sub_LatticeColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourMatrix _sub_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.lexico_layout.shape())
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

cdef class ColourVector:
    cdef colour_vector.ColourVector* instance
    cdef int view_count
    cdef Py_ssize_t buffer_shape[1]
    cdef Py_ssize_t buffer_strides[1]

    def __cinit__(self, ):
        self.view_count = 0
        self.instance = new colour_vector.ColourVector(colour_vector.zeros())

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = 3
        self.buffer_strides[0] = itemsize

        buffer.buf = <char*>self.instance
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 3
        buffer.ndim = 1

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer* buffer):
        pass

    property as_numpy:
        """Return a view to this object as a numpy array"""
        def __get__(self):
            out = np.asarray(self)
            out.dtype = complex
            return out

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out[:] = value

    def __add__(self, other):
        if type(self) is ColourVector and type(other) is ColourVector:
            return (<ColourVector>self)._add_ColourVector_ColourVector(<ColourVector>other)
        raise TypeError("Unsupported operand types for ColourVector.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourVector _add_ColourVector_ColourVector(ColourVector self, ColourVector other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        raise TypeError("Unsupported operand types for ColourVector.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    def __sub__(self, other):
        if type(self) is ColourVector and type(other) is ColourVector:
            return (<ColourVector>self)._sub_ColourVector_ColourVector(<ColourVector>other)
        raise TypeError("Unsupported operand types for ColourVector.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourVector _sub_ColourVector_ColourVector(ColourVector self, ColourVector other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        raise TypeError("Unsupported operand types for ColourVector.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        raise TypeError("Unsupported operand types for ColourVector.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

cdef class LatticeColourVector:
    cdef lattice_colour_vector.LatticeColourVector* instance
    cdef layout.Layout* lexico_layout
    cdef int view_count
    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]

    def __cinit__(self, shape):
        self.lexico_layout = new layout.LexicoLayout(shape)
        self.view_count = 0
        self.instance = new lattice_colour_vector.LatticeColourVector(self.lexico_layout[0], colour_vector.ColourVector(colour_vector.zeros()))

    def __dealloc__(self):
        del self.instance
        del self.lexico_layout

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance[0].volume()
        self.buffer_strides[0] = itemsize * 3
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 3 * self.instance[0].volume()
        buffer.ndim = 2

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1
    property as_numpy:
        """Return a view to this object as a numpy array"""
        def __get__(self):
            out = np.asarray(self)
            out.dtype = complex
            return out

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out[:] = value

    def __add__(self, other):
        if type(self) is LatticeColourVector and type(other) is LatticeColourVector:
            return (<LatticeColourVector>self)._add_LatticeColourVector_LatticeColourVector(<LatticeColourVector>other)
        raise TypeError("Unsupported operand types for LatticeColourVector.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourVector _add_LatticeColourVector_LatticeColourVector(LatticeColourVector self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.lexico_layout.shape())
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        raise TypeError("Unsupported operand types for LatticeColourVector.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    def __sub__(self, other):
        if type(self) is LatticeColourVector and type(other) is LatticeColourVector:
            return (<LatticeColourVector>self)._sub_LatticeColourVector_LatticeColourVector(<LatticeColourVector>other)
        raise TypeError("Unsupported operand types for LatticeColourVector.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourVector _sub_LatticeColourVector_LatticeColourVector(LatticeColourVector self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.lexico_layout.shape())
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        raise TypeError("Unsupported operand types for LatticeColourVector.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        raise TypeError("Unsupported operand types for LatticeColourVector.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

