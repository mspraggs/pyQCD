from cpython cimport Py_buffer
from libcpp.vector cimport vector

from mpi4py import MPI
import numpy as np

cimport comms
from comms cimport MPI_Comm
cimport complex
from core cimport layout
from core cimport ColourMatrix, LatticeColourMatrix, ColourVector, LatticeColourVector

def init_comms(comm):
    cdef size_t comm_ptr = <size_t>MPI._addressof(comm)
    comms.Communicator.instance().init((<MPI_Comm*>comm_ptr)[0])


cdef class Layout:

    def __cinit__(self, shape, partition, halo_depth = 1, max_mpi_hop = 1):
        self.instance = new layout.Layout(shape, partition, halo_depth,
                                          max_mpi_hop)


cdef class ColourMatrix:

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
            return out.reshape((3, 3))

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out = out.reshape((3, 3))
            out[:] = value

    def __add__(self, other):
        if type(self) is ColourMatrix and type(other) is ColourMatrix:
            return (<ColourMatrix>self)._add_ColourMatrix_ColourMatrix(<ColourMatrix>other)
        raise TypeError("Unsupported operand types for ColourMatrix.__add__: "
                        "{} and {}".format(type(self), type(other)))

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

    def __sub__(self, other):
        if type(self) is ColourMatrix and type(other) is ColourMatrix:
            return (<ColourMatrix>self)._sub_ColourMatrix_ColourMatrix(<ColourMatrix>other)
        raise TypeError("Unsupported operand types for ColourMatrix.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    def __div__(self, other):
        raise TypeError("Unsupported operand types for ColourMatrix.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        raise TypeError("Unsupported operand types for ColourMatrix.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

cdef class LatticeColourMatrix:

    def __cinit__(self, layout, int site_size=1):
        self.layout = layout
        self.view_count = 0
        self.site_size = site_size
        self.instance = new lattice_colour_matrix.LatticeColourMatrix(self.layout.instance[0], colour_matrix.ColourMatrix(colour_matrix.zeros()), site_size)

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance[0].local_volume() * self.site_size
        self.buffer_strides[0] = itemsize * 9
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize
        self.buffer_shape[2] = 3
        self.buffer_strides[2] = itemsize * 3

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 9 * self.instance[0].local_volume() * self.site_size
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
            return out.reshape(tuple(self.layout.instance.local_shape()) + (self.site_size,) + (3, 3))

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out = out.reshape(tuple(self.layout.instance.local_shape()) + (self.site_size,) + (3, 3))
            out[:] = value

    def __add__(self, other):
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>self)._add_LatticeColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__add__: "
                        "{} and {}".format(type(self), type(other)))

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

    def __sub__(self, other):
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>self)._sub_LatticeColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    def __div__(self, other):
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

cdef class ColourVector:

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
            return out.reshape((3,))

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out = out.reshape((3,))
            out[:] = value

    def __add__(self, other):
        if type(self) is ColourVector and type(other) is ColourVector:
            return (<ColourVector>self)._add_ColourVector_ColourVector(<ColourVector>other)
        raise TypeError("Unsupported operand types for ColourVector.__add__: "
                        "{} and {}".format(type(self), type(other)))

    def __mul__(self, other):
        raise TypeError("Unsupported operand types for ColourVector.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    def __sub__(self, other):
        if type(self) is ColourVector and type(other) is ColourVector:
            return (<ColourVector>self)._sub_ColourVector_ColourVector(<ColourVector>other)
        raise TypeError("Unsupported operand types for ColourVector.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    def __div__(self, other):
        raise TypeError("Unsupported operand types for ColourVector.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        raise TypeError("Unsupported operand types for ColourVector.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

cdef class LatticeColourVector:

    def __cinit__(self, layout, int site_size=1):
        self.layout = layout
        self.view_count = 0
        self.site_size = site_size
        self.instance = new lattice_colour_vector.LatticeColourVector(self.layout.instance[0], colour_vector.ColourVector(colour_vector.zeros()), site_size)

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance[0].local_volume() * self.site_size
        self.buffer_strides[0] = itemsize * 3
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * 3 * self.instance[0].local_volume() * self.site_size
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
            return out.reshape(tuple(self.layout.instance.local_shape()) + (self.site_size,) + (3,))

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out = out.reshape(tuple(self.layout.instance.local_shape()) + (self.site_size,) + (3,))
            out[:] = value

    def __add__(self, other):
        if type(self) is LatticeColourVector and type(other) is LatticeColourVector:
            return (<LatticeColourVector>self)._add_LatticeColourVector_LatticeColourVector(<LatticeColourVector>other)
        raise TypeError("Unsupported operand types for LatticeColourVector.__add__: "
                        "{} and {}".format(type(self), type(other)))

    def __mul__(self, other):
        raise TypeError("Unsupported operand types for LatticeColourVector.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    def __sub__(self, other):
        if type(self) is LatticeColourVector and type(other) is LatticeColourVector:
            return (<LatticeColourVector>self)._sub_LatticeColourVector_LatticeColourVector(<LatticeColourVector>other)
        raise TypeError("Unsupported operand types for LatticeColourVector.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    def __div__(self, other):
        raise TypeError("Unsupported operand types for LatticeColourVector.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        raise TypeError("Unsupported operand types for LatticeColourVector.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

