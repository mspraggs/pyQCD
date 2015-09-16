from cpython cimport Py_buffer
from libcpp.vector cimport vector

import numpy as np

cimport complex
cimport layout
from operators cimport *
cimport colour_matrix
cimport colour_matrix_array
cimport lattice_colour_matrix
cimport gauge_field
cimport colour_vector
cimport fermion
cimport lattice_colour_vector
cimport fermion_field

scalar_types = (int, float, np.single, np.double,
                np.float16, np.float32, np.float64, np.float128)
complex_types = (complex, np.complex, np.complex64, np.complex128,
                 np.complex256)

ctypedef double Real

cdef class Complex:
    cdef complex.Complex instance

    def __init__(self, x, y):
        self.instance = complex.Complex(x, y)

    @property
    def real(self):
        return self.instance.real()

    @property
    def imag(self):
        return self.instance.imag()

    def to_complex(self):
        return complex(self.instance.real(), self.instance.imag())

    def __repr__(self):
        if self.real == 0:
            return "{}j".format(self.imag)
        else:
            return "({}{}{}j)".format(self.real,
                                      ('+' if self.imag >= 0 else ''),
                                      self.imag)


cdef class Layout:
    cdef layout.Layout* instance

    def get_array_index(self, site_label):

        if type(site_label) is int:
            return self.instance.get_array_index(<unsigned int>site_label)
        if type(site_label) is list or type(site_label) is tuple:
            return self.instance.get_array_index(<vector[unsigned int]>site_label)
        raise TypeError("Unknown type in Layout.get_array_index: {}"
                        .format(type(site_label)))

    def get_site_index(self, unsigned int array_index):
        return self.instance.get_site_index(array_index)

    @property
    def num_dims(self):
        return self.instance.num_dims()

    @property
    def volume(self):
        return self.instance.volume()

    @property
    def lattice_shape(self):
        return tuple(self.instance.lattice_shape())


cdef class LexicoLayout(Layout):

    def __init__(self, shape):
        self.instance = <layout.Layout*>new layout.LexicoLayout(<vector[unsigned int]?>shape)

    def __dealloc__(self):
        del self.instance


cdef class ColourMatrix:
    cdef colour_matrix.ColourMatrix* instance
    cdef Layout layout

    def __cinit__(self, ):
        self.instance = new colour_matrix.ColourMatrix(colour_matrix.zeros())

    def __init__(self, ):
        self.instance = new colour_matrix.ColourMatrix(colour_matrix.zeros())

    def __dealloc__(self):
        del self.instance

    cdef int view_count
    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]

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

    @property
    def as_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

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
        if type(self) is ColourMatrix and type(other) is ColourMatrixArray:
            return (<ColourMatrix>self)._mul_ColourMatrix_ColourMatrixArray(<ColourMatrixArray>other)
        if type(self) is ColourMatrix and type(other) is ColourVector:
            return (<ColourMatrix>self)._mul_ColourMatrix_ColourVector(<ColourVector>other)
        if type(self) is ColourMatrix and type(other) is Fermion:
            return (<ColourMatrix>self)._mul_ColourMatrix_Fermion(<Fermion>other)
        if type(self) is complex and type(other) is ColourMatrix:
            return (<ColourMatrix>other)._mul_ColourMatrix_Complex(Complex(self.real, self.imag))
        if type(self) is ColourMatrix and type(other) is complex:
            return (<ColourMatrix>self)._mul_ColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrix.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrix _mul_ColourMatrix_ColourMatrix(ColourMatrix self, ColourMatrix other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline ColourMatrixArray _mul_ColourMatrix_ColourMatrixArray(ColourMatrix self, ColourMatrixArray other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline ColourVector _mul_ColourMatrix_ColourVector(ColourMatrix self, ColourVector other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline Fermion _mul_ColourMatrix_Fermion(ColourMatrix self, Fermion other):
        cdef Fermion out = Fermion()
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
        if type(self) is complex and type(other) is ColourMatrix:
            return (<ColourMatrix>other)._div_ColourMatrix_Complex(Complex(self.real, self.imag))
        if type(self) is ColourMatrix and type(other) is complex:
            return (<ColourMatrix>self)._div_ColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrix.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is complex and type(other) is ColourMatrix:
            return (<ColourMatrix>other)._div_ColourMatrix_Complex(Complex(self.real, self.imag))
        if type(self) is ColourMatrix and type(other) is complex:
            return (<ColourMatrix>self)._div_ColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrix.__truediv__: "
                        "{} and {}".format(type(self), type(other)))


cdef class ColourMatrixArray:
    cdef colour_matrix_array.ColourMatrixArray* instance
    cdef Layout layout

    def __cinit__(self, int size):
        self.instance = new colour_matrix_array.ColourMatrixArray(size, colour_matrix.ColourMatrix(colour_matrix.zeros()))

    def __init__(self, int size):
        self.instance = new colour_matrix_array.ColourMatrixArray(size, colour_matrix.ColourMatrix(colour_matrix.zeros()))

    def __dealloc__(self):
        del self.instance

    cdef int view_count
    cdef Py_ssize_t buffer_shape[3]
    cdef Py_ssize_t buffer_strides[3]

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance[0].size()
        self.buffer_strides[0] = itemsize * 9
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize
        self.buffer_shape[2] = 3
        self.buffer_strides[2] = itemsize * 3

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 9 * self.instance[0].size()
        buffer.ndim = 3

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1
    @property
    def as_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    def __add__(self, other):
        if type(self) is ColourMatrixArray and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>self)._add_ColourMatrixArray_ColourMatrixArray(<ColourMatrixArray>other)
        raise TypeError("Unsupported operand types for ColourMatrixArray.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrixArray _add_ColourMatrixArray_ColourMatrixArray(ColourMatrixArray self, ColourMatrixArray other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is ColourMatrixArray and type(other) is ColourMatrix:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_ColourMatrix(<ColourMatrix>other)
        if type(self) is ColourMatrixArray and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_ColourMatrixArray(<ColourMatrixArray>other)
        if type(self) is ColourMatrixArray and type(other) is ColourVector:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_ColourVector(<ColourVector>other)
        if type(self) is ColourMatrixArray and type(other) is Fermion:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_Fermion(<Fermion>other)
        if type(self) is complex and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>other)._mul_ColourMatrixArray_Complex(Complex(self.real, self.imag))
        if type(self) is ColourMatrixArray and type(other) is complex:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrixArray.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrixArray _mul_ColourMatrixArray_ColourMatrix(ColourMatrixArray self, ColourMatrix other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline ColourMatrixArray _mul_ColourMatrixArray_ColourMatrixArray(ColourMatrixArray self, ColourMatrixArray other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline Fermion _mul_ColourMatrixArray_ColourVector(ColourMatrixArray self, ColourVector other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline Fermion _mul_ColourMatrixArray_Fermion(ColourMatrixArray self, Fermion other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    def __sub__(self, other):
        if type(self) is ColourMatrixArray and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>self)._sub_ColourMatrixArray_ColourMatrixArray(<ColourMatrixArray>other)
        raise TypeError("Unsupported operand types for ColourMatrixArray.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrixArray _sub_ColourMatrixArray_ColourMatrixArray(ColourMatrixArray self, ColourMatrixArray other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        if type(self) is complex and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>other)._div_ColourMatrixArray_Complex(Complex(self.real, self.imag))
        if type(self) is ColourMatrixArray and type(other) is complex:
            return (<ColourMatrixArray>self)._div_ColourMatrixArray_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrixArray.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is complex and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>other)._div_ColourMatrixArray_Complex(Complex(self.real, self.imag))
        if type(self) is ColourMatrixArray and type(other) is complex:
            return (<ColourMatrixArray>self)._div_ColourMatrixArray_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrixArray.__truediv__: "
                        "{} and {}".format(type(self), type(other)))


cdef class LatticeColourMatrix:
    cdef lattice_colour_matrix.LatticeColourMatrix* instance
    cdef Layout layout

    def __cinit__(self, Layout layout):
        self.instance = new lattice_colour_matrix.LatticeColourMatrix(layout.instance[0], colour_matrix.ColourMatrix(colour_matrix.zeros()))

    def __init__(self, Layout layout):
        self.instance = new lattice_colour_matrix.LatticeColourMatrix(layout.instance[0], colour_matrix.ColourMatrix(colour_matrix.zeros()))

    def __dealloc__(self):
        del self.instance

    cdef int view_count
    cdef Py_ssize_t buffer_shape[3]
    cdef Py_ssize_t buffer_strides[3]

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
    @property
    def as_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    def __add__(self, other):
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>self)._add_LatticeColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourMatrix _add_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        if type(self) is LatticeColourMatrix and type(other) is GaugeField:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_GaugeField(<GaugeField>other)
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourVector:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_LatticeColourVector(<LatticeColourVector>other)
        if type(self) is LatticeColourMatrix and type(other) is FermionField:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_FermionField(<FermionField>other)
        if type(self) is complex and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>other)._mul_LatticeColourMatrix_Complex(Complex(self.real, self.imag))
        if type(self) is LatticeColourMatrix and type(other) is complex:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourMatrix _mul_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline GaugeField _mul_LatticeColourMatrix_GaugeField(LatticeColourMatrix self, GaugeField other):
        cdef GaugeField out = GaugeField(self.layout)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourVector _mul_LatticeColourMatrix_LatticeColourVector(LatticeColourMatrix self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline FermionField _mul_LatticeColourMatrix_FermionField(LatticeColourMatrix self, FermionField other):
        cdef FermionField out = FermionField(self.layout)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    def __sub__(self, other):
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>self)._sub_LatticeColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourMatrix _sub_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        if type(self) is complex and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>other)._div_LatticeColourMatrix_Complex(Complex(self.real, self.imag))
        if type(self) is LatticeColourMatrix and type(other) is complex:
            return (<LatticeColourMatrix>self)._div_LatticeColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is complex and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>other)._div_LatticeColourMatrix_Complex(Complex(self.real, self.imag))
        if type(self) is LatticeColourMatrix and type(other) is complex:
            return (<LatticeColourMatrix>self)._div_LatticeColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__truediv__: "
                        "{} and {}".format(type(self), type(other)))


cdef class GaugeField:
    cdef gauge_field.GaugeField* instance
    cdef Layout layout

    def __cinit__(self, Layout layout, int size):
        self.instance = new gauge_field.GaugeField(layout.instance[0], colour_matrix_array.ColourMatrixArray(size, colour_matrix.ColourMatrix(colour_matrix.zeros())))

    def __init__(self, Layout layout, int size):
        self.instance = new gauge_field.GaugeField(layout.instance[0], colour_matrix_array.ColourMatrixArray(size, colour_matrix.ColourMatrix(colour_matrix.zeros())))

    def __dealloc__(self):
        del self.instance

    cdef int view_count
    cdef Py_ssize_t buffer_shape[4]
    cdef Py_ssize_t buffer_strides[4]

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance[0].volume()
        self.buffer_strides[0] = itemsize * 9 * self.instance[0][0].size()
        self.buffer_shape[1] = self.instance[0][0].size()
        self.buffer_strides[1] = itemsize * 9
        self.buffer_shape[2] = 3
        self.buffer_strides[2] = itemsize
        self.buffer_shape[3] = 3
        self.buffer_strides[3] = itemsize * 3

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 9 * self.instance[0][0].size() * self.instance[0].volume()
        buffer.ndim = 4

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1
    @property
    def as_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    def __add__(self, other):
        if type(self) is GaugeField and type(other) is GaugeField:
            return (<GaugeField>self)._add_GaugeField_GaugeField(<GaugeField>other)
        raise TypeError("Unsupported operand types for GaugeField.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline GaugeField _add_GaugeField_GaugeField(GaugeField self, GaugeField other):
        cdef GaugeField out = GaugeField(self.layout)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is GaugeField and type(other) is LatticeColourMatrix:
            return (<GaugeField>self)._mul_GaugeField_LatticeColourMatrix(<LatticeColourMatrix>other)
        if type(self) is GaugeField and type(other) is GaugeField:
            return (<GaugeField>self)._mul_GaugeField_GaugeField(<GaugeField>other)
        if type(self) is GaugeField and type(other) is LatticeColourVector:
            return (<GaugeField>self)._mul_GaugeField_LatticeColourVector(<LatticeColourVector>other)
        if type(self) is GaugeField and type(other) is FermionField:
            return (<GaugeField>self)._mul_GaugeField_FermionField(<FermionField>other)
        if type(self) is complex and type(other) is GaugeField:
            return (<GaugeField>other)._mul_GaugeField_Complex(Complex(self.real, self.imag))
        if type(self) is GaugeField and type(other) is complex:
            return (<GaugeField>self)._mul_GaugeField_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for GaugeField.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline GaugeField _mul_GaugeField_LatticeColourMatrix(GaugeField self, LatticeColourMatrix other):
        cdef GaugeField out = GaugeField(self.layout)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline GaugeField _mul_GaugeField_GaugeField(GaugeField self, GaugeField other):
        cdef GaugeField out = GaugeField(self.layout)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline FermionField _mul_GaugeField_LatticeColourVector(GaugeField self, LatticeColourVector other):
        cdef FermionField out = FermionField(self.layout)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline FermionField _mul_GaugeField_FermionField(GaugeField self, FermionField other):
        cdef FermionField out = FermionField(self.layout)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    def __sub__(self, other):
        if type(self) is GaugeField and type(other) is GaugeField:
            return (<GaugeField>self)._sub_GaugeField_GaugeField(<GaugeField>other)
        raise TypeError("Unsupported operand types for GaugeField.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline GaugeField _sub_GaugeField_GaugeField(GaugeField self, GaugeField other):
        cdef GaugeField out = GaugeField(self.layout)
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        if type(self) is complex and type(other) is GaugeField:
            return (<GaugeField>other)._div_GaugeField_Complex(Complex(self.real, self.imag))
        if type(self) is GaugeField and type(other) is complex:
            return (<GaugeField>self)._div_GaugeField_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for GaugeField.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is complex and type(other) is GaugeField:
            return (<GaugeField>other)._div_GaugeField_Complex(Complex(self.real, self.imag))
        if type(self) is GaugeField and type(other) is complex:
            return (<GaugeField>self)._div_GaugeField_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for GaugeField.__truediv__: "
                        "{} and {}".format(type(self), type(other)))


cdef class ColourVector:
    cdef colour_vector.ColourVector* instance
    cdef Layout layout

    def __cinit__(self, ):
        self.instance = new colour_vector.ColourVector(colour_vector.zeros())

    def __init__(self, ):
        self.instance = new colour_vector.ColourVector(colour_vector.zeros())

    def __dealloc__(self):
        del self.instance

    cdef int view_count
    cdef Py_ssize_t buffer_shape[1]
    cdef Py_ssize_t buffer_strides[1]

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

    @property
    def as_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

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
        if type(self) is complex and type(other) is ColourVector:
            return (<ColourVector>other)._mul_ColourVector_Complex(Complex(self.real, self.imag))
        if type(self) is ColourVector and type(other) is complex:
            return (<ColourVector>self)._mul_ColourVector_Complex(Complex(other.real, other.imag))
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
        if type(self) is complex and type(other) is ColourVector:
            return (<ColourVector>other)._div_ColourVector_Complex(Complex(self.real, self.imag))
        if type(self) is ColourVector and type(other) is complex:
            return (<ColourVector>self)._div_ColourVector_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourVector.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is complex and type(other) is ColourVector:
            return (<ColourVector>other)._div_ColourVector_Complex(Complex(self.real, self.imag))
        if type(self) is ColourVector and type(other) is complex:
            return (<ColourVector>self)._div_ColourVector_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourVector.__truediv__: "
                        "{} and {}".format(type(self), type(other)))


cdef class Fermion:
    cdef fermion.Fermion* instance
    cdef Layout layout

    def __cinit__(self, int size):
        self.instance = new fermion.Fermion(size, colour_vector.ColourVector(colour_vector.zeros()))

    def __init__(self, int size):
        self.instance = new fermion.Fermion(size, colour_vector.ColourVector(colour_vector.zeros()))

    def __dealloc__(self):
        del self.instance

    cdef int view_count
    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance[0].size()
        self.buffer_strides[0] = itemsize * 3
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 3 * self.instance[0].size()
        buffer.ndim = 2

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1
    @property
    def as_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    def __add__(self, other):
        if type(self) is Fermion and type(other) is Fermion:
            return (<Fermion>self)._add_Fermion_Fermion(<Fermion>other)
        raise TypeError("Unsupported operand types for Fermion.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline Fermion _add_Fermion_Fermion(Fermion self, Fermion other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is complex and type(other) is Fermion:
            return (<Fermion>other)._mul_Fermion_Complex(Complex(self.real, self.imag))
        if type(self) is Fermion and type(other) is complex:
            return (<Fermion>self)._mul_Fermion_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for Fermion.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    def __sub__(self, other):
        if type(self) is Fermion and type(other) is Fermion:
            return (<Fermion>self)._sub_Fermion_Fermion(<Fermion>other)
        raise TypeError("Unsupported operand types for Fermion.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline Fermion _sub_Fermion_Fermion(Fermion self, Fermion other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        if type(self) is complex and type(other) is Fermion:
            return (<Fermion>other)._div_Fermion_Complex(Complex(self.real, self.imag))
        if type(self) is Fermion and type(other) is complex:
            return (<Fermion>self)._div_Fermion_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for Fermion.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is complex and type(other) is Fermion:
            return (<Fermion>other)._div_Fermion_Complex(Complex(self.real, self.imag))
        if type(self) is Fermion and type(other) is complex:
            return (<Fermion>self)._div_Fermion_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for Fermion.__truediv__: "
                        "{} and {}".format(type(self), type(other)))


cdef class LatticeColourVector:
    cdef lattice_colour_vector.LatticeColourVector* instance
    cdef Layout layout

    def __cinit__(self, Layout layout):
        self.instance = new lattice_colour_vector.LatticeColourVector(layout.instance[0], colour_vector.ColourVector(colour_vector.zeros()))

    def __init__(self, Layout layout):
        self.instance = new lattice_colour_vector.LatticeColourVector(layout.instance[0], colour_vector.ColourVector(colour_vector.zeros()))

    def __dealloc__(self):
        del self.instance

    cdef int view_count
    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]

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
    @property
    def as_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    def __add__(self, other):
        if type(self) is LatticeColourVector and type(other) is LatticeColourVector:
            return (<LatticeColourVector>self)._add_LatticeColourVector_LatticeColourVector(<LatticeColourVector>other)
        raise TypeError("Unsupported operand types for LatticeColourVector.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourVector _add_LatticeColourVector_LatticeColourVector(LatticeColourVector self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is complex and type(other) is LatticeColourVector:
            return (<LatticeColourVector>other)._mul_LatticeColourVector_Complex(Complex(self.real, self.imag))
        if type(self) is LatticeColourVector and type(other) is complex:
            return (<LatticeColourVector>self)._mul_LatticeColourVector_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourVector.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    def __sub__(self, other):
        if type(self) is LatticeColourVector and type(other) is LatticeColourVector:
            return (<LatticeColourVector>self)._sub_LatticeColourVector_LatticeColourVector(<LatticeColourVector>other)
        raise TypeError("Unsupported operand types for LatticeColourVector.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourVector _sub_LatticeColourVector_LatticeColourVector(LatticeColourVector self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        if type(self) is complex and type(other) is LatticeColourVector:
            return (<LatticeColourVector>other)._div_LatticeColourVector_Complex(Complex(self.real, self.imag))
        if type(self) is LatticeColourVector and type(other) is complex:
            return (<LatticeColourVector>self)._div_LatticeColourVector_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourVector.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is complex and type(other) is LatticeColourVector:
            return (<LatticeColourVector>other)._div_LatticeColourVector_Complex(Complex(self.real, self.imag))
        if type(self) is LatticeColourVector and type(other) is complex:
            return (<LatticeColourVector>self)._div_LatticeColourVector_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourVector.__truediv__: "
                        "{} and {}".format(type(self), type(other)))


cdef class FermionField:
    cdef fermion_field.FermionField* instance
    cdef Layout layout

    def __cinit__(self, Layout layout, int size):
        self.instance = new fermion_field.FermionField(layout.instance[0], fermion.Fermion(size, colour_vector.ColourVector(colour_vector.zeros())))

    def __init__(self, Layout layout, int size):
        self.instance = new fermion_field.FermionField(layout.instance[0], fermion.Fermion(size, colour_vector.ColourVector(colour_vector.zeros())))

    def __dealloc__(self):
        del self.instance

    cdef int view_count
    cdef Py_ssize_t buffer_shape[3]
    cdef Py_ssize_t buffer_strides[3]

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance[0].volume()
        self.buffer_strides[0] = itemsize * 3 * self.instance[0][0].size()
        self.buffer_shape[1] = self.instance[0][0].size()
        self.buffer_strides[1] = itemsize * 3
        self.buffer_shape[2] = 3
        self.buffer_strides[2] = itemsize

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 3 * self.instance[0][0].size() * self.instance[0].volume()
        buffer.ndim = 3

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1
    @property
    def as_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    def __add__(self, other):
        if type(self) is FermionField and type(other) is FermionField:
            return (<FermionField>self)._add_FermionField_FermionField(<FermionField>other)
        raise TypeError("Unsupported operand types for FermionField.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline FermionField _add_FermionField_FermionField(FermionField self, FermionField other):
        cdef FermionField out = FermionField(self.layout)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is complex and type(other) is FermionField:
            return (<FermionField>other)._mul_FermionField_Complex(Complex(self.real, self.imag))
        if type(self) is FermionField and type(other) is complex:
            return (<FermionField>self)._mul_FermionField_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for FermionField.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    def __sub__(self, other):
        if type(self) is FermionField and type(other) is FermionField:
            return (<FermionField>self)._sub_FermionField_FermionField(<FermionField>other)
        raise TypeError("Unsupported operand types for FermionField.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline FermionField _sub_FermionField_FermionField(FermionField self, FermionField other):
        cdef FermionField out = FermionField(self.layout)
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        if type(self) is complex and type(other) is FermionField:
            return (<FermionField>other)._div_FermionField_Complex(Complex(self.real, self.imag))
        if type(self) is FermionField and type(other) is complex:
            return (<FermionField>self)._div_FermionField_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for FermionField.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is complex and type(other) is FermionField:
            return (<FermionField>other)._div_FermionField_Complex(Complex(self.real, self.imag))
        if type(self) is FermionField and type(other) is complex:
            return (<FermionField>self)._div_FermionField_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for FermionField.__truediv__: "
                        "{} and {}".format(type(self), type(other)))


