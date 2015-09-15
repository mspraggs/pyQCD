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

    def __cinit__(self, ):
        self.instance = new colour_matrix.ColourMatrix(colour_matrix.zeros())

    def __init__(self, ):
        self.instance = new colour_matrix.ColourMatrix(colour_matrix.zeros())

    def __dealloc__(self):
        del self.instance

    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]
    cdef int view_count

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
cdef class ColourMatrixArray:
    cdef colour_matrix_array.ColourMatrixArray* instance

    def __cinit__(self, int size):
        self.instance = new colour_matrix_array.ColourMatrixArray(size, colour_matrix.ColourMatrix(colour_matrix.zeros()))

    def __init__(self, int size):
        self.instance = new colour_matrix_array.ColourMatrixArray(size, colour_matrix.ColourMatrix(colour_matrix.zeros()))

    def __dealloc__(self):
        del self.instance

    cdef Py_ssize_t buffer_shape[3]
    cdef Py_ssize_t buffer_strides[3]
    cdef int view_count

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
cdef class LatticeColourMatrix:
    cdef lattice_colour_matrix.LatticeColourMatrix* instance

    def __cinit__(self, Layout layout):
        self.instance = new lattice_colour_matrix.LatticeColourMatrix(layout.instance[0], colour_matrix.ColourMatrix(colour_matrix.zeros()))

    def __init__(self, Layout layout):
        self.instance = new lattice_colour_matrix.LatticeColourMatrix(layout.instance[0], colour_matrix.ColourMatrix(colour_matrix.zeros()))

    def __dealloc__(self):
        del self.instance

    cdef Py_ssize_t buffer_shape[3]
    cdef Py_ssize_t buffer_strides[3]
    cdef int view_count

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
cdef class GaugeField:
    cdef gauge_field.GaugeField* instance

    def __cinit__(self, Layout layout, int size):
        self.instance = new gauge_field.GaugeField(layout.instance[0], colour_matrix_array.ColourMatrixArray(size, colour_matrix.ColourMatrix(colour_matrix.zeros())))

    def __init__(self, Layout layout, int size):
        self.instance = new gauge_field.GaugeField(layout.instance[0], colour_matrix_array.ColourMatrixArray(size, colour_matrix.ColourMatrix(colour_matrix.zeros())))

    def __dealloc__(self):
        del self.instance

    cdef Py_ssize_t buffer_shape[4]
    cdef Py_ssize_t buffer_strides[4]
    cdef int view_count

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
cdef class ColourVector:
    cdef colour_vector.ColourVector* instance

    def __cinit__(self, ):
        self.instance = new colour_vector.ColourVector(colour_vector.zeros())

    def __init__(self, ):
        self.instance = new colour_vector.ColourVector(colour_vector.zeros())

    def __dealloc__(self):
        del self.instance

    cdef Py_ssize_t buffer_shape[1]
    cdef Py_ssize_t buffer_strides[1]
    cdef int view_count

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
cdef class Fermion:
    cdef fermion.Fermion* instance

    def __cinit__(self, int size):
        self.instance = new fermion.Fermion(size, colour_vector.ColourVector(colour_vector.zeros()))

    def __init__(self, int size):
        self.instance = new fermion.Fermion(size, colour_vector.ColourVector(colour_vector.zeros()))

    def __dealloc__(self):
        del self.instance

    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]
    cdef int view_count

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
cdef class LatticeColourVector:
    cdef lattice_colour_vector.LatticeColourVector* instance

    def __cinit__(self, Layout layout):
        self.instance = new lattice_colour_vector.LatticeColourVector(layout.instance[0], colour_vector.ColourVector(colour_vector.zeros()))

    def __init__(self, Layout layout):
        self.instance = new lattice_colour_vector.LatticeColourVector(layout.instance[0], colour_vector.ColourVector(colour_vector.zeros()))

    def __dealloc__(self):
        del self.instance

    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]
    cdef int view_count

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
cdef class FermionField:
    cdef fermion_field.FermionField* instance

    def __cinit__(self, Layout layout, int size):
        self.instance = new fermion_field.FermionField(layout.instance[0], fermion.Fermion(size, colour_vector.ColourVector(colour_vector.zeros())))

    def __init__(self, Layout layout, int size):
        self.instance = new fermion_field.FermionField(layout.instance[0], fermion.Fermion(size, colour_vector.ColourVector(colour_vector.zeros())))

    def __dealloc__(self):
        del self.instance

    cdef Py_ssize_t buffer_shape[3]
    cdef Py_ssize_t buffer_strides[3]
    cdef int view_count

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
