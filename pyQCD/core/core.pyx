from cpython cimport Py_buffer
from libcpp.vector cimport vector

import numpy as np

cimport atomics
cimport core
from core cimport _ColourMatrix, ColourMatrix, _LatticeColourMatrix, LatticeColourMatrix, _ColourVector, ColourVector, _LatticeColourVector, LatticeColourVector


cdef class ColourMatrix:

    def __cinit__(self):
        self.instance = new _ColourMatrix(core._ColourMatrix_zeros())
        self.view_count = 0

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(atomics.Complex)

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

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1

    property as_numpy:
        """Return a view to this object as a numpy array"""
        def __get__(self):
            out = np.asarray(self)
            out.dtype = complex
            return out.reshape((3, 3))

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out = out.reshape((3, 3))
            out[:] = value

cdef class LatticeColourMatrix:

    def __cinit__(self, shape, int site_size=1):
        self.lexico_layout = new LexicoLayout(shape)
        self.view_count = 0
        self.site_size = site_size
        self.instance = new _LatticeColourMatrix(self.lexico_layout[0],  _ColourMatrix(_ColourMatrix_zeros()), site_size)

    def __dealloc__(self):
        del self.instance
        del self.lexico_layout

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(atomics.Complex)

        self.buffer_shape[0] = self.instance[0].volume() * self.site_size
        self.buffer_strides[0] = itemsize * 9
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize
        self.buffer_shape[2] = 3
        self.buffer_strides[2] = itemsize * 3

        buffer.buf = <char*>&(self.instance[0][0])

        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 9 * self.instance[0].volume() * self.site_size
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
            return out.reshape(tuple(self.lexico_layout.shape()) + (self.site_size,) + (3, 3))

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out = out.reshape(tuple(self.lexico_layout.shape()) + (self.site_size,) + (3, 3))
            out[:] = value

cdef class ColourVector:

    def __cinit__(self):
        self.instance = new _ColourVector(core._ColourVector_zeros())
        self.view_count = 0

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(atomics.Complex)

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

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1

    property as_numpy:
        """Return a view to this object as a numpy array"""
        def __get__(self):
            out = np.asarray(self)
            out.dtype = complex
            return out.reshape((3,))

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out = out.reshape((3,))
            out[:] = value

cdef class LatticeColourVector:

    def __cinit__(self, shape, int site_size=1):
        self.lexico_layout = new LexicoLayout(shape)
        self.view_count = 0
        self.site_size = site_size
        self.instance = new _LatticeColourVector(self.lexico_layout[0],  _ColourVector(_ColourVector_zeros()), site_size)

    def __dealloc__(self):
        del self.instance
        del self.lexico_layout

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(atomics.Complex)

        self.buffer_shape[0] = self.instance[0].volume() * self.site_size
        self.buffer_strides[0] = itemsize * 3
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize

        buffer.buf = <char*>&(self.instance[0][0])

        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 3 * self.instance[0].volume() * self.site_size
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
            return out.reshape(tuple(self.lexico_layout.shape()) + (self.site_size,) + (3,))

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out = out.reshape(tuple(self.lexico_layout.shape()) + (self.site_size,) + (3,))
            out[:] = value
