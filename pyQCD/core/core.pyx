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


cdef inline int validate_ColourMatrix_indices(int i, int j) except -1:
    if i > 2 or i < 0 or j > 2 or j < 0:
        raise IndexError("Indices in ColourMatrix element access out of bounds: "
                         "{}".format((i, j)))


cdef inline int validate_LatticeColourMatrix_index(index, vector[unsigned int] shape, unsigned int num_dims, unsigned int volume) except -1:
    cdef int i
    if type(index) is tuple:
        for i in range(num_dims):
            if index[i] >= shape[i] or index[i] < 0:
                raise IndexError("Index in LatticeColourMatrix element access "
                                 "out of bounds: {}".format(index))
    elif type(index) is int:
        if index < 0 or index >= volume:
            raise IndexError("Index in LatticeColourMatrix element access "
                             "out of bounds: {}".format(index))


cdef class ColourMatrix:
    cdef colour_matrix.ColourMatrix* instance
    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]
    shape = (3, 3)

    cdef colour_matrix.ColourMatrix cppobj(self):
        return self.instance[0]

    cdef validate_indices(self, int i, int j ):
        validate_ColourMatrix_indices(i, j)

    def __cinit__(self):
        self.instance = new colour_matrix.ColourMatrix()

    def __init__(self, *args):
        cdef int i, j
        if not args:
            pass
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            for i, elem in enumerate(args[0]):
                for j, subelem in enumerate(elem):
                    self.validate_indices(i, j)
                    self[i, j] = subelem

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = 3
        self.buffer_strides[0] = itemsize
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = 3 * itemsize

        buffer.buf = <char*>self.instance
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = 9 * 1 * itemsize
        buffer.ndim = 2

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer* buffer):
        pass

    def __getitem__(self, index):
        out = Complex(0.0, 0.0)
        if type(index) is tuple:
            self.validate_indices(index[0], index[1])
            out.instance = self.instance[0](index[0], index[1])
        else:
            raise TypeError("Invalid index type in ColourMatrix.__setitem__: "
                            "{}".format(type(index)))
        return out.to_complex()

    def __setitem__(self, index, value):
        if type(value) is Complex:
            pass
        elif hasattr(value, 'real') and hasattr(value, 'imag'):
            value = Complex(<double?>(value.real),
                            <double?>(value.imag))
        else:
            value = Complex(<double?>value, 0.0)
        if type(index) is tuple:
            self.validate_indices(index[0], index[1])
            self.assign_elem(index[0], index[1], (<Complex>value).instance)
        else:
            raise TypeError("Invalid index type in ColourMatrix.__setitem__: "
                            "{}".format(type(index)))

    cdef void assign_elem(self, int i, int j, complex.Complex value):
        colour_matrix.mat_assign(self.instance, i, j, value)

    def adjoint(self):
        out = ColourMatrix()
        out.instance[0] = self.instance[0].adjoint()
        return out

    @staticmethod
    def zeros():
        out = ColourMatrix()
        out.instance[0] = colour_matrix.zeros()
        return out

    @staticmethod
    def ones():
        out = ColourMatrix()
        out.instance[0] = colour_matrix.ones()
        return out

    @staticmethod
    def identity():
        out = ColourMatrix()
        out.instance[0] = colour_matrix.identity()
        return out

    def to_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    def __add__(self, other):
        if type(self) is ColourMatrix and type(other) is ColourMatrix:
            return (<ColourMatrix>self)._add_ColourMatrix_ColourMatrix(<ColourMatrix>other)
        if type(self) is ColourMatrix and type(other) is ColourMatrixArray:
            return (<ColourMatrix>self)._add_ColourMatrix_ColourMatrixArray(<ColourMatrixArray>other)
        if type(self) is ColourMatrix and type(other) is LatticeColourMatrix:
            return (<ColourMatrix>self)._add_ColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        if type(self) is ColourMatrix and type(other) is GaugeField:
            return (<ColourMatrix>self)._add_ColourMatrix_GaugeField(<GaugeField>other)
        raise TypeError("Unsupported operand types for ColourMatrix.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrix _add_ColourMatrix_ColourMatrix(ColourMatrix self, ColourMatrix other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline ColourMatrixArray _add_ColourMatrix_ColourMatrixArray(ColourMatrix self, ColourMatrixArray other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline LatticeColourMatrix _add_ColourMatrix_LatticeColourMatrix(ColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline GaugeField _add_ColourMatrix_GaugeField(ColourMatrix self, GaugeField other):
        cdef GaugeField out = GaugeField()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is float and type(other) is ColourMatrix:
            return (<ColourMatrix>other)._mul_ColourMatrix_float(<float>self)
        if type(self) is ColourMatrix and type(other) is float:
            return (<ColourMatrix>self)._mul_ColourMatrix_float(<float>other)
        if type(self) is int and type(other) is ColourMatrix:
            return (<ColourMatrix>other)._mul_ColourMatrix_int(<int>self)
        if type(self) is ColourMatrix and type(other) is int:
            return (<ColourMatrix>self)._mul_ColourMatrix_int(<int>other)
        if type(self) is Complex and type(other) is ColourMatrix:
            return (<ColourMatrix>other)._mul_ColourMatrix_Complex(<Complex>self)
        if type(self) is ColourMatrix and type(other) is Complex:
            return (<ColourMatrix>self)._mul_ColourMatrix_Complex(<Complex>other)
        if type(self) is ColourMatrix and type(other) is ColourMatrix:
            return (<ColourMatrix>self)._mul_ColourMatrix_ColourMatrix(<ColourMatrix>other)
        if type(self) is ColourMatrix and type(other) is ColourMatrixArray:
            return (<ColourMatrix>self)._mul_ColourMatrix_ColourMatrixArray(<ColourMatrixArray>other)
        if type(self) is ColourMatrix and type(other) is LatticeColourMatrix:
            return (<ColourMatrix>self)._mul_ColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        if type(self) is ColourMatrix and type(other) is GaugeField:
            return (<ColourMatrix>self)._mul_ColourMatrix_GaugeField(<GaugeField>other)
        if type(self) is ColourMatrix and type(other) is ColourVector:
            return (<ColourMatrix>self)._mul_ColourMatrix_ColourVector(<ColourVector>other)
        if type(self) is ColourMatrix and type(other) is Fermion:
            return (<ColourMatrix>self)._mul_ColourMatrix_Fermion(<Fermion>other)
        if type(self) is ColourMatrix and type(other) is LatticeColourVector:
            return (<ColourMatrix>self)._mul_ColourMatrix_LatticeColourVector(<LatticeColourVector>other)
        if type(self) is ColourMatrix and type(other) is FermionField:
            return (<ColourMatrix>self)._mul_ColourMatrix_FermionField(<FermionField>other)
        if hasattr(self, "real") and hasattr(self, "imag") and type(other) is ColourMatrix:
            return (<ColourMatrix>other)._mul_ColourMatrix_Complex(Complex(self.real, self.imag))
        if type(self) is ColourMatrix and hasattr(other, "real") and hasattr(other, "imag"):
            return (<ColourMatrix>self)._mul_ColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrix.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrix _mul_ColourMatrix_float(ColourMatrix self, float other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline ColourMatrix _mul_ColourMatrix_int(ColourMatrix self, int other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline ColourMatrix _mul_ColourMatrix_Complex(ColourMatrix self, Complex other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] * other.instance
        return out

    cdef inline ColourMatrix _mul_ColourMatrix_ColourMatrix(ColourMatrix self, ColourMatrix other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline ColourMatrixArray _mul_ColourMatrix_ColourMatrixArray(ColourMatrix self, ColourMatrixArray other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourMatrix _mul_ColourMatrix_LatticeColourMatrix(ColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline GaugeField _mul_ColourMatrix_GaugeField(ColourMatrix self, GaugeField other):
        cdef GaugeField out = GaugeField()
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

    cdef inline LatticeColourVector _mul_ColourMatrix_LatticeColourVector(ColourMatrix self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector()
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline FermionField _mul_ColourMatrix_FermionField(ColourMatrix self, FermionField other):
        cdef FermionField out = FermionField()
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
        if type(self) is ColourMatrix and type(other) is float:
            return (<ColourMatrix>self)._div_ColourMatrix_float(<float>other)
        if type(self) is ColourMatrix and type(other) is int:
            return (<ColourMatrix>self)._div_ColourMatrix_int(<int>other)
        if type(self) is ColourMatrix and type(other) is Complex:
            return (<ColourMatrix>self)._div_ColourMatrix_Complex(<Complex>other)
        if type(self) is ColourMatrix and hasattr(other, "real") and hasattr(other, "imag"):
            return (<ColourMatrix>self)._div_ColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrix.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is ColourMatrix and type(other) is float:
            return (<ColourMatrix>self)._div_ColourMatrix_float(<float>other)
        if type(self) is ColourMatrix and type(other) is int:
            return (<ColourMatrix>self)._div_ColourMatrix_int(<int>other)
        if type(self) is ColourMatrix and type(other) is Complex:
            return (<ColourMatrix>self)._div_ColourMatrix_Complex(<Complex>other)
        if type(self) is ColourMatrix and hasattr(other, "real") and hasattr(other, "imag"):
            return (<ColourMatrix>self)._div_ColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrix.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrix _div_ColourMatrix_float(ColourMatrix self, float other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline ColourMatrix _div_ColourMatrix_int(ColourMatrix self, int other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline ColourMatrix _div_ColourMatrix_Complex(ColourMatrix self, Complex other):
        cdef ColourMatrix out = ColourMatrix()
        out.instance[0] = self.instance[0] / other.instance
        return out


cdef class ColourMatrixArray:
    cdef colour_matrix_array.ColourMatrixArray* instance
    cdef Py_ssize_t buffer_shape[3]
    cdef Py_ssize_t buffer_strides[3]
    cdef int view_count

    cdef colour_matrix_array.ColourMatrixArray cppobj(self):
        return self.instance[0]

    cdef _init_with_args_(self, unsigned int N, ColourMatrix value):
        self.instance[0] = colour_matrix_array.ColourMatrixArray(N, value.instance[0])

    cdef validate_index(self, int i):
        if i >= self.instance.size() or i < 0:
            raise IndexError("Index in ColourMatrixArray element access out of bounds: "
                             "{}".format(i))

    def __cinit__(self):
        self.instance = new colour_matrix_array.ColourMatrixArray()
        self.view_count = 0

    def __init__(self, *args):
        cdef int i, N
        if not args:
            pass
        elif len(args) == 1 and hasattr(args[0], "__len__"):
            N = len(args[0])
            self.instance.resize(N)
            for i in range(N):
                self[i] = ColourMatrix(args[0][i])
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], ColourMatrix):
            self._init_with_args_(args[0], args[1])
        else:
            raise TypeError("ColourMatrixArray constructor expects "
                            "either zero or two arguments")

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance.size()
        self.buffer_strides[0] = 9 * itemsize
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize
        self.buffer_shape[2] = 3
        self.buffer_strides[2] = 3 * itemsize

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = 9 * self.instance.size() * itemsize
        buffer.ndim = 3

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1

    def __getitem__(self, index):
        if type(index) is tuple and len(index) is 1:
            self.validate_index(index[0])
            out = ColourMatrix()
            (<ColourMatrix>out).instance[0] = (self.instance[0])[<int?>(index[0])]
            return out
        elif type(index) is tuple:
            self.validate_index(index[0])
            out = Complex(0.0, 0.0)
            validate_ColourMatrix_indices(index[1], index[2])
            (<Complex>out).instance = self.instance[0][<int?>index[0]](<int?>index[1], <int?>index[2])
            return out.to_complex()
        else:
            self.validate_index(index)
            out = ColourMatrix()
            (<ColourMatrix>out).instance[0] = self.instance[0][<int?>index]
            return out

    def __setitem__(self, index, value):
        if type(value) is ColourMatrix:
            self.validate_index(index[0] if type(index) is tuple else index)
            self.assign_elem(index[0] if type(index) is tuple else index, (<ColourMatrix>value).instance[0])
            return
        elif type(value) is Complex:
            pass
        elif hasattr(value, "real") and hasattr(value, "imag") and isinstance(index, tuple):
            value = Complex(value.real, value.imag)
        else:
            value = Complex(<double?>value, 0.0)

        cdef colour_matrix.ColourMatrix* mat = &(self.instance[0][<int?>index[0]])
        validate_ColourMatrix_indices(index[1], index[2])
        colour_matrix.mat_assign(mat, <int?>index[1], <int?>index[2], (<Complex?>value).instance)

    cdef void assign_elem(self, int i, colour_matrix.ColourMatrix value):
        cdef colour_matrix.ColourMatrix* m = &(self.instance[0][i])
        m[0] = value

    def adjoint(self):
        out = ColourMatrixArray()
        out.instance[0] = self.instance[0].adjoint()
        return out

    @staticmethod
    def zeros(int num_elements):
        out = ColourMatrixArray()
        out.instance[0] = colour_matrix_array.ColourMatrixArray(num_elements, colour_matrix.zeros())
        return out

    @staticmethod
    def ones(int num_elements):
        out = ColourMatrixArray()
        out.instance[0] = colour_matrix_array.ColourMatrixArray(num_elements, colour_matrix.ones())
        return out

    @staticmethod
    def identity(int num_elements):
        out = ColourMatrixArray()
        out.instance[0] = colour_matrix_array.ColourMatrixArray(num_elements, colour_matrix.identity())
        return out

    def to_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    @property
    def size(self):
        return self.instance.size()

    @property
    def shape(self):
        return (self.size, 3, 3)

    def __add__(self, other):
        if type(self) is ColourMatrixArray and type(other) is ColourMatrix:
            return (<ColourMatrixArray>self)._add_ColourMatrixArray_ColourMatrix(<ColourMatrix>other)
        if type(self) is ColourMatrixArray and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>self)._add_ColourMatrixArray_ColourMatrixArray(<ColourMatrixArray>other)
        raise TypeError("Unsupported operand types for ColourMatrixArray.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrixArray _add_ColourMatrixArray_ColourMatrix(ColourMatrixArray self, ColourMatrix other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline ColourMatrixArray _add_ColourMatrixArray_ColourMatrixArray(ColourMatrixArray self, ColourMatrixArray other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is float and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>other)._mul_ColourMatrixArray_float(<float>self)
        if type(self) is ColourMatrixArray and type(other) is float:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_float(<float>other)
        if type(self) is int and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>other)._mul_ColourMatrixArray_int(<int>self)
        if type(self) is ColourMatrixArray and type(other) is int:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_int(<int>other)
        if type(self) is Complex and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>other)._mul_ColourMatrixArray_Complex(<Complex>self)
        if type(self) is ColourMatrixArray and type(other) is Complex:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_Complex(<Complex>other)
        if type(self) is ColourMatrixArray and type(other) is ColourMatrix:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_ColourMatrix(<ColourMatrix>other)
        if type(self) is ColourMatrixArray and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_ColourMatrixArray(<ColourMatrixArray>other)
        if type(self) is ColourMatrixArray and type(other) is ColourVector:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_ColourVector(<ColourVector>other)
        if type(self) is ColourMatrixArray and type(other) is Fermion:
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_Fermion(<Fermion>other)
        if hasattr(self, "real") and hasattr(self, "imag") and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>other)._mul_ColourMatrixArray_Complex(Complex(self.real, self.imag))
        if type(self) is ColourMatrixArray and hasattr(other, "real") and hasattr(other, "imag"):
            return (<ColourMatrixArray>self)._mul_ColourMatrixArray_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrixArray.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrixArray _mul_ColourMatrixArray_float(ColourMatrixArray self, float other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline ColourMatrixArray _mul_ColourMatrixArray_int(ColourMatrixArray self, int other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline ColourMatrixArray _mul_ColourMatrixArray_Complex(ColourMatrixArray self, Complex other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] * other.instance
        return out

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
        if type(self) is ColourMatrixArray and type(other) is ColourMatrix:
            return (<ColourMatrixArray>self)._sub_ColourMatrixArray_ColourMatrix(<ColourMatrix>other)
        if type(self) is ColourMatrixArray and type(other) is ColourMatrixArray:
            return (<ColourMatrixArray>self)._sub_ColourMatrixArray_ColourMatrixArray(<ColourMatrixArray>other)
        raise TypeError("Unsupported operand types for ColourMatrixArray.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrixArray _sub_ColourMatrixArray_ColourMatrix(ColourMatrixArray self, ColourMatrix other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    cdef inline ColourMatrixArray _sub_ColourMatrixArray_ColourMatrixArray(ColourMatrixArray self, ColourMatrixArray other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        if type(self) is ColourMatrixArray and type(other) is float:
            return (<ColourMatrixArray>self)._div_ColourMatrixArray_float(<float>other)
        if type(self) is ColourMatrixArray and type(other) is int:
            return (<ColourMatrixArray>self)._div_ColourMatrixArray_int(<int>other)
        if type(self) is ColourMatrixArray and type(other) is Complex:
            return (<ColourMatrixArray>self)._div_ColourMatrixArray_Complex(<Complex>other)
        if type(self) is ColourMatrixArray and hasattr(other, "real") and hasattr(other, "imag"):
            return (<ColourMatrixArray>self)._div_ColourMatrixArray_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrixArray.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is ColourMatrixArray and type(other) is float:
            return (<ColourMatrixArray>self)._div_ColourMatrixArray_float(<float>other)
        if type(self) is ColourMatrixArray and type(other) is int:
            return (<ColourMatrixArray>self)._div_ColourMatrixArray_int(<int>other)
        if type(self) is ColourMatrixArray and type(other) is Complex:
            return (<ColourMatrixArray>self)._div_ColourMatrixArray_Complex(<Complex>other)
        if type(self) is ColourMatrixArray and hasattr(other, "real") and hasattr(other, "imag"):
            return (<ColourMatrixArray>self)._div_ColourMatrixArray_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourMatrixArray.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourMatrixArray _div_ColourMatrixArray_float(ColourMatrixArray self, float other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline ColourMatrixArray _div_ColourMatrixArray_int(ColourMatrixArray self, int other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline ColourMatrixArray _div_ColourMatrixArray_Complex(ColourMatrixArray self, Complex other):
        cdef ColourMatrixArray out = ColourMatrixArray()
        out.instance[0] = self.instance[0] / other.instance
        return out


cdef class LatticeColourMatrix:
    cdef lattice_colour_matrix.LatticeColourMatrix* instance
    cdef Layout layout
    cdef Py_ssize_t buffer_shape[3]
    cdef Py_ssize_t buffer_strides[3]
    cdef int view_count

    cdef lattice_colour_matrix.LatticeColourMatrix cppobj(self):
        return self.instance[0]

    cdef validate_index(self, index):
        validate_LatticeColourMatrix_index(
            index, self.instance.lattice_shape(), self.instance.num_dims(),
            self.instance.volume())

    def __cinit__(self, Layout layout, *args):
        self.instance = new lattice_colour_matrix.LatticeColourMatrix(layout.instance[0], colour_matrix.ColourMatrix())
        self.layout = layout
        self.view_count = 0

    def __init__(self, Layout layout, *args):
        cdef int i, volume
        volume = layout.instance.volume()
        if len(args) is 1 and type(args[0]) is ColourMatrix:
            for i in range(volume):
                self.instance[0][i] = (<ColourMatrix>args[0]).instance[0]

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance.volume()
        self.buffer_strides[0] = 9 * itemsize
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize
        self.buffer_shape[2] = 3
        self.buffer_strides[2] = 3 * itemsize

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = 9 * self.instance.volume() * itemsize
        buffer.ndim = 3

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1

    def __getitem__(self, index):
        cdef int num_dims = self.instance.num_dims()
        if type(index) is tuple and len(index) == self.instance.num_dims():
            out = ColourMatrix()
            self.validate_index(index)
            (<ColourMatrix>out).instance[0] = (<LatticeColourMatrix>self).instance[0](<vector[unsigned int]>index)
            return out
        if type(index) is tuple and len(index) == num_dims + 2:
            out = Complex(0.0, 0.0)
            self.validate_index(index)
            validate_ColourMatrix_indices(index[num_dims], index[num_dims + 1])
            (<Complex>out).instance = (<LatticeColourMatrix>self).instance[0](<vector[unsigned int]>index[:num_dims])(index[num_dims], index[num_dims + 1])
            return out.to_complex()
        if type(index) is int:
            out = ColourMatrix()
            self.validate_index(index)
            (<ColourMatrix>out).instance[0] = (<LatticeColourMatrix>self).instance[0](<int>index)
            return out
        raise TypeError("Invalid index type in LatticeColourMatrix.__getitem__")

    def __setitem__(self, index, value):
        cdef int num_dims = self.instance.num_dims()
        if type(value) is ColourMatrix:
            self.validate_index(index[:num_dims] if type(index) is tuple else index)
            self.assign_elem(index[:num_dims] if type(index) is tuple else index, (<ColourMatrix>value).instance[0])
            return
        elif type(value) is Complex:
            pass
        elif hasattr(value, "real") and hasattr(value, "imag") and isinstance(index, tuple):
            value = Complex(value.real, value.imag)
        else:
            value = Complex(<double?>value, 0.0)

        cdef colour_matrix.ColourMatrix* mat
        if type(index) is tuple:
            mat = &(self.instance[0](<vector[unsigned int]?>index[:num_dims]))
        else:
            mat = &(self.instance[0](<int?>index))
        validate_ColourMatrix_indices(index[num_dims], index[num_dims + 1])
        colour_matrix.mat_assign(mat, <int?>index[num_dims], <int?>index[num_dims + 1], (<Complex?>value).instance)

    cdef assign_elem(self, index, colour_matrix.ColourMatrix value):
        cdef colour_matrix.ColourMatrix* m
        if type(index) is tuple:
            m = &(self.instance[0](<vector[unsigned int]>index))
        else:
            m = &(self.instance[0](<int?>index))
        m[0] = value

    def adjoint(self):
        raise NotImplementedError
        out = LatticeColourMatrix()
        out.instance[0] = self.instance[0].adjoint()
        return out

    @staticmethod
    def zeros(Layout layout):
        return LatticeColourMatrix(layout, ColourMatrix.zeros())

    @staticmethod
    def ones(Layout layout):
        return LatticeColourMatrix(layout, ColourMatrix.ones())

    @staticmethod
    def identity(Layout layout):
        return LatticeColourMatrix(layout, ColourMatrix.identity())
    
    def to_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    @property
    def num_dims(self):
        return self.instance.num_dims()

    @property
    def volume(self):
        return self.instance.volume()

    @property
    def lattice_shape(self):
        return tuple(self.instance.lattice_shape())

    @property
    def shape(self):
        return tuple(self.instance.lattice_shape()) + ColourMatrix.shape

    def __add__(self, other):
        if type(self) is LatticeColourMatrix and type(other) is ColourMatrix:
            return (<LatticeColourMatrix>self)._add_LatticeColourMatrix_ColourMatrix(<ColourMatrix>other)
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>self)._add_LatticeColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        if type(self) is LatticeColourMatrix and type(other) is GaugeField:
            return (<LatticeColourMatrix>self)._add_LatticeColourMatrix_GaugeField(<GaugeField>other)
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourMatrix _add_LatticeColourMatrix_ColourMatrix(LatticeColourMatrix self, ColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline LatticeColourMatrix _add_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline GaugeField _add_LatticeColourMatrix_GaugeField(LatticeColourMatrix self, GaugeField other):
        cdef GaugeField out = GaugeField(self.layout)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is float and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>other)._mul_LatticeColourMatrix_float(<float>self)
        if type(self) is LatticeColourMatrix and type(other) is float:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_float(<float>other)
        if type(self) is int and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>other)._mul_LatticeColourMatrix_int(<int>self)
        if type(self) is LatticeColourMatrix and type(other) is int:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_int(<int>other)
        if type(self) is Complex and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>other)._mul_LatticeColourMatrix_Complex(<Complex>self)
        if type(self) is LatticeColourMatrix and type(other) is Complex:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_Complex(<Complex>other)
        if type(self) is LatticeColourMatrix and type(other) is ColourMatrix:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_ColourMatrix(<ColourMatrix>other)
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        if type(self) is LatticeColourMatrix and type(other) is GaugeField:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_GaugeField(<GaugeField>other)
        if type(self) is LatticeColourMatrix and type(other) is ColourVector:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_ColourVector(<ColourVector>other)
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourVector:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_LatticeColourVector(<LatticeColourVector>other)
        if type(self) is LatticeColourMatrix and type(other) is FermionField:
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_FermionField(<FermionField>other)
        if hasattr(self, "real") and hasattr(self, "imag") and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>other)._mul_LatticeColourMatrix_Complex(Complex(self.real, self.imag))
        if type(self) is LatticeColourMatrix and hasattr(other, "real") and hasattr(other, "imag"):
            return (<LatticeColourMatrix>self)._mul_LatticeColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourMatrix _mul_LatticeColourMatrix_float(LatticeColourMatrix self, float other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline LatticeColourMatrix _mul_LatticeColourMatrix_int(LatticeColourMatrix self, int other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline LatticeColourMatrix _mul_LatticeColourMatrix_Complex(LatticeColourMatrix self, Complex other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] * other.instance
        return out

    cdef inline LatticeColourMatrix _mul_LatticeColourMatrix_ColourMatrix(LatticeColourMatrix self, ColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourMatrix _mul_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline GaugeField _mul_LatticeColourMatrix_GaugeField(LatticeColourMatrix self, GaugeField other):
        cdef GaugeField out = GaugeField(self.layout)
        out.instance[0] = self.instance[0] * other.instance[0]
        return out

    cdef inline LatticeColourVector _mul_LatticeColourMatrix_ColourVector(LatticeColourMatrix self, ColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
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
        if type(self) is LatticeColourMatrix and type(other) is ColourMatrix:
            return (<LatticeColourMatrix>self)._sub_LatticeColourMatrix_ColourMatrix(<ColourMatrix>other)
        if type(self) is LatticeColourMatrix and type(other) is LatticeColourMatrix:
            return (<LatticeColourMatrix>self)._sub_LatticeColourMatrix_LatticeColourMatrix(<LatticeColourMatrix>other)
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourMatrix _sub_LatticeColourMatrix_ColourMatrix(LatticeColourMatrix self, ColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    cdef inline LatticeColourMatrix _sub_LatticeColourMatrix_LatticeColourMatrix(LatticeColourMatrix self, LatticeColourMatrix other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        if type(self) is LatticeColourMatrix and type(other) is float:
            return (<LatticeColourMatrix>self)._div_LatticeColourMatrix_float(<float>other)
        if type(self) is LatticeColourMatrix and type(other) is int:
            return (<LatticeColourMatrix>self)._div_LatticeColourMatrix_int(<int>other)
        if type(self) is LatticeColourMatrix and type(other) is Complex:
            return (<LatticeColourMatrix>self)._div_LatticeColourMatrix_Complex(<Complex>other)
        if type(self) is LatticeColourMatrix and hasattr(other, "real") and hasattr(other, "imag"):
            return (<LatticeColourMatrix>self)._div_LatticeColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is LatticeColourMatrix and type(other) is float:
            return (<LatticeColourMatrix>self)._div_LatticeColourMatrix_float(<float>other)
        if type(self) is LatticeColourMatrix and type(other) is int:
            return (<LatticeColourMatrix>self)._div_LatticeColourMatrix_int(<int>other)
        if type(self) is LatticeColourMatrix and type(other) is Complex:
            return (<LatticeColourMatrix>self)._div_LatticeColourMatrix_Complex(<Complex>other)
        if type(self) is LatticeColourMatrix and hasattr(other, "real") and hasattr(other, "imag"):
            return (<LatticeColourMatrix>self)._div_LatticeColourMatrix_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourMatrix.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourMatrix _div_LatticeColourMatrix_float(LatticeColourMatrix self, float other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline LatticeColourMatrix _div_LatticeColourMatrix_int(LatticeColourMatrix self, int other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline LatticeColourMatrix _div_LatticeColourMatrix_Complex(LatticeColourMatrix self, Complex other):
        cdef LatticeColourMatrix out = LatticeColourMatrix(self.layout)
        out.instance[0] = self.instance[0] / other.instance
        return out


cdef class GaugeField:
    cdef gauge_field.GaugeField* instance
    def __init__(self):
        pass


cdef inline int validate_ColourVector_indices(int i) except -1:
    if i > 2 or i < 0:
        raise IndexError("Indices in ColourVector element access out of bounds: "
                         "{}".format((i)))


cdef inline int validate_LatticeColourVector_index(index, vector[unsigned int] shape, unsigned int num_dims, unsigned int volume) except -1:
    cdef int i
    if type(index) is tuple:
        for i in range(num_dims):
            if index[i] >= shape[i] or index[i] < 0:
                raise IndexError("Index in LatticeColourVector element access "
                                 "out of bounds: {}".format(index))
    elif type(index) is int:
        if index < 0 or index >= volume:
            raise IndexError("Index in LatticeColourVector element access "
                             "out of bounds: {}".format(index))


cdef class ColourVector:
    cdef colour_vector.ColourVector* instance
    cdef Py_ssize_t buffer_shape[1]
    cdef Py_ssize_t buffer_strides[1]
    shape = (3,)

    cdef colour_vector.ColourVector cppobj(self):
        return self.instance[0]

    cdef validate_indices(self, int i):
        validate_ColourVector_indices(i)

    def __cinit__(self):
        self.instance = new colour_vector.ColourVector()

    def __init__(self, *args):
        cdef int i, j
        if not args:
            pass
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            for i, elem in enumerate(args[0]):
                self.validate_indices(i)
                self[i] = elem

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
        buffer.len = 3 * 1 * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer* buffer):
        pass

    def __getitem__(self, index):
        out = Complex(0.0, 0.0)
        if type(index) is tuple:
            self.validate_indices(index[0])
            out.instance = self.instance[0][index[0]]
        elif type(index) is int:
            self.validate_indices(index)
            out.instance = self.instance[0][index]
        else:
            raise TypeError("Invalid index type in ColourVector.__setitem__: "
                            "{}".format(type(index)))
        return out.to_complex()

    def __setitem__(self, index, value):
        if type(value) is Complex:
            pass
        elif hasattr(value, 'real') and hasattr(value, 'imag'):
            value = Complex(<double?>(value.real),
                            <double?>(value.imag))
        else:
            value = Complex(<double?>value, 0.0)
        if type(index) is tuple:
            self.validate_indices(index[0])
            self.assign_elem(index[0], (<Complex>value).instance)
        elif type(index) is int:
            self.validate_indices(index)
            self.assign_elem(index, (<Complex>value).instance)
        else:
            raise TypeError("Invalid index type in ColourVector.__setitem__: "
                            "{}".format(type(index)))

    cdef void assign_elem(self, int i, complex.Complex value):
        cdef complex.Complex* z = &(self.instance[0][i])
        z[0] = value

    def adjoint(self):
        out = ColourVector()
        out.instance[0] = self.instance[0].adjoint()
        return out

    @staticmethod
    def zeros():
        out = ColourVector()
        out.instance[0] = colour_vector.zeros()
        return out

    @staticmethod
    def ones():
        out = ColourVector()
        out.instance[0] = colour_vector.ones()
        return out

    def to_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    def __add__(self, other):
        if type(self) is ColourVector and type(other) is ColourVector:
            return (<ColourVector>self)._add_ColourVector_ColourVector(<ColourVector>other)
        if type(self) is ColourVector and type(other) is Fermion:
            return (<ColourVector>self)._add_ColourVector_Fermion(<Fermion>other)
        if type(self) is ColourVector and type(other) is LatticeColourVector:
            return (<ColourVector>self)._add_ColourVector_LatticeColourVector(<LatticeColourVector>other)
        if type(self) is ColourVector and type(other) is FermionField:
            return (<ColourVector>self)._add_ColourVector_FermionField(<FermionField>other)
        raise TypeError("Unsupported operand types for ColourVector.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourVector _add_ColourVector_ColourVector(ColourVector self, ColourVector other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline Fermion _add_ColourVector_Fermion(ColourVector self, Fermion other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline LatticeColourVector _add_ColourVector_LatticeColourVector(ColourVector self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline FermionField _add_ColourVector_FermionField(ColourVector self, FermionField other):
        cdef FermionField out = FermionField()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is float and type(other) is ColourVector:
            return (<ColourVector>other)._mul_ColourVector_float(<float>self)
        if type(self) is ColourVector and type(other) is float:
            return (<ColourVector>self)._mul_ColourVector_float(<float>other)
        if type(self) is int and type(other) is ColourVector:
            return (<ColourVector>other)._mul_ColourVector_int(<int>self)
        if type(self) is ColourVector and type(other) is int:
            return (<ColourVector>self)._mul_ColourVector_int(<int>other)
        if type(self) is Complex and type(other) is ColourVector:
            return (<ColourVector>other)._mul_ColourVector_Complex(<Complex>self)
        if type(self) is ColourVector and type(other) is Complex:
            return (<ColourVector>self)._mul_ColourVector_Complex(<Complex>other)
        if hasattr(self, "real") and hasattr(self, "imag") and type(other) is ColourVector:
            return (<ColourVector>other)._mul_ColourVector_Complex(Complex(self.real, self.imag))
        if type(self) is ColourVector and hasattr(other, "real") and hasattr(other, "imag"):
            return (<ColourVector>self)._mul_ColourVector_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourVector.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourVector _mul_ColourVector_float(ColourVector self, float other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline ColourVector _mul_ColourVector_int(ColourVector self, int other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline ColourVector _mul_ColourVector_Complex(ColourVector self, Complex other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] * other.instance
        return out

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
        if type(self) is ColourVector and type(other) is float:
            return (<ColourVector>self)._div_ColourVector_float(<float>other)
        if type(self) is ColourVector and type(other) is int:
            return (<ColourVector>self)._div_ColourVector_int(<int>other)
        if type(self) is ColourVector and type(other) is Complex:
            return (<ColourVector>self)._div_ColourVector_Complex(<Complex>other)
        if type(self) is ColourVector and hasattr(other, "real") and hasattr(other, "imag"):
            return (<ColourVector>self)._div_ColourVector_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourVector.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is ColourVector and type(other) is float:
            return (<ColourVector>self)._div_ColourVector_float(<float>other)
        if type(self) is ColourVector and type(other) is int:
            return (<ColourVector>self)._div_ColourVector_int(<int>other)
        if type(self) is ColourVector and type(other) is Complex:
            return (<ColourVector>self)._div_ColourVector_Complex(<Complex>other)
        if type(self) is ColourVector and hasattr(other, "real") and hasattr(other, "imag"):
            return (<ColourVector>self)._div_ColourVector_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for ColourVector.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline ColourVector _div_ColourVector_float(ColourVector self, float other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline ColourVector _div_ColourVector_int(ColourVector self, int other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline ColourVector _div_ColourVector_Complex(ColourVector self, Complex other):
        cdef ColourVector out = ColourVector()
        out.instance[0] = self.instance[0] / other.instance
        return out


cdef class Fermion:
    cdef fermion.Fermion* instance
    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]
    cdef int view_count

    cdef fermion.Fermion cppobj(self):
        return self.instance[0]

    cdef _init_with_args_(self, unsigned int N, ColourVector value):
        self.instance[0] = fermion.Fermion(N, value.instance[0])

    cdef validate_index(self, int i):
        if i >= self.instance.size() or i < 0:
            raise IndexError("Index in Fermion element access out of bounds: "
                             "{}".format(i))

    def __cinit__(self):
        self.instance = new fermion.Fermion()
        self.view_count = 0

    def __init__(self, *args):
        cdef int i, N
        if not args:
            pass
        elif len(args) == 1 and hasattr(args[0], "__len__"):
            N = len(args[0])
            self.instance.resize(N)
            for i in range(N):
                self[i] = ColourVector(args[0][i])
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], ColourVector):
            self._init_with_args_(args[0], args[1])
        else:
            raise TypeError("Fermion constructor expects "
                            "either zero or two arguments")

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance.size()
        self.buffer_strides[0] = 3 * itemsize
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = 3 * self.instance.size() * itemsize
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1

    def __getitem__(self, index):
        if type(index) is tuple and len(index) is 1:
            self.validate_index(index[0])
            out = ColourVector()
            (<ColourVector>out).instance[0] = (self.instance[0])[<int?>(index[0])]
            return out
        elif type(index) is tuple:
            self.validate_index(index[0])
            out = Complex(0.0, 0.0)
            validate_ColourVector_indices(index[1])
            (<Complex>out).instance = self.instance[0][<int?>index[0]][<int?>index[1]]
            return out.to_complex()
        else:
            self.validate_index(index)
            out = ColourVector()
            (<ColourVector>out).instance[0] = self.instance[0][<int?>index]
            return out

    def __setitem__(self, index, value):
        if type(value) is ColourVector:
            self.validate_index(index[0] if type(index) is tuple else index)
            self.assign_elem(index[0] if type(index) is tuple else index, (<ColourVector>value).instance[0])
            return
        elif type(value) is Complex:
            pass
        elif hasattr(value, "real") and hasattr(value, "imag") and isinstance(index, tuple):
            value = Complex(value.real, value.imag)
        else:
            value = Complex(<double?>value, 0.0)

        cdef colour_vector.ColourVector* mat = &(self.instance[0][<int?>index[0]])
        validate_ColourVector_indices(index[1])
        cdef complex.Complex* z = &(mat[0][<int?>index[1]])
        z[0] = (<Complex>value).instance

    cdef void assign_elem(self, int i, colour_vector.ColourVector value):
        cdef colour_vector.ColourVector* m = &(self.instance[0][i])
        m[0] = value

    def adjoint(self):
        out = Fermion()
        out.instance[0] = self.instance[0].adjoint()
        return out

    @staticmethod
    def zeros(int num_elements):
        out = Fermion()
        out.instance[0] = fermion.Fermion(num_elements, colour_vector.zeros())
        return out

    @staticmethod
    def ones(int num_elements):
        out = Fermion()
        out.instance[0] = fermion.Fermion(num_elements, colour_vector.ones())
        return out

    def to_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    @property
    def size(self):
        return self.instance.size()

    @property
    def shape(self):
        return (self.size, 3,)

    def __add__(self, other):
        if type(self) is Fermion and type(other) is ColourVector:
            return (<Fermion>self)._add_Fermion_ColourVector(<ColourVector>other)
        if type(self) is Fermion and type(other) is Fermion:
            return (<Fermion>self)._add_Fermion_Fermion(<Fermion>other)
        raise TypeError("Unsupported operand types for Fermion.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline Fermion _add_Fermion_ColourVector(Fermion self, ColourVector other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline Fermion _add_Fermion_Fermion(Fermion self, Fermion other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is float and type(other) is Fermion:
            return (<Fermion>other)._mul_Fermion_float(<float>self)
        if type(self) is Fermion and type(other) is float:
            return (<Fermion>self)._mul_Fermion_float(<float>other)
        if type(self) is int and type(other) is Fermion:
            return (<Fermion>other)._mul_Fermion_int(<int>self)
        if type(self) is Fermion and type(other) is int:
            return (<Fermion>self)._mul_Fermion_int(<int>other)
        if type(self) is Complex and type(other) is Fermion:
            return (<Fermion>other)._mul_Fermion_Complex(<Complex>self)
        if type(self) is Fermion and type(other) is Complex:
            return (<Fermion>self)._mul_Fermion_Complex(<Complex>other)
        if hasattr(self, "real") and hasattr(self, "imag") and type(other) is Fermion:
            return (<Fermion>other)._mul_Fermion_Complex(Complex(self.real, self.imag))
        if type(self) is Fermion and hasattr(other, "real") and hasattr(other, "imag"):
            return (<Fermion>self)._mul_Fermion_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for Fermion.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline Fermion _mul_Fermion_float(Fermion self, float other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline Fermion _mul_Fermion_int(Fermion self, int other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline Fermion _mul_Fermion_Complex(Fermion self, Complex other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] * other.instance
        return out

    def __sub__(self, other):
        if type(self) is Fermion and type(other) is ColourVector:
            return (<Fermion>self)._sub_Fermion_ColourVector(<ColourVector>other)
        if type(self) is Fermion and type(other) is Fermion:
            return (<Fermion>self)._sub_Fermion_Fermion(<Fermion>other)
        raise TypeError("Unsupported operand types for Fermion.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline Fermion _sub_Fermion_ColourVector(Fermion self, ColourVector other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    cdef inline Fermion _sub_Fermion_Fermion(Fermion self, Fermion other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        if type(self) is Fermion and type(other) is float:
            return (<Fermion>self)._div_Fermion_float(<float>other)
        if type(self) is Fermion and type(other) is int:
            return (<Fermion>self)._div_Fermion_int(<int>other)
        if type(self) is Fermion and type(other) is Complex:
            return (<Fermion>self)._div_Fermion_Complex(<Complex>other)
        if type(self) is Fermion and hasattr(other, "real") and hasattr(other, "imag"):
            return (<Fermion>self)._div_Fermion_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for Fermion.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is Fermion and type(other) is float:
            return (<Fermion>self)._div_Fermion_float(<float>other)
        if type(self) is Fermion and type(other) is int:
            return (<Fermion>self)._div_Fermion_int(<int>other)
        if type(self) is Fermion and type(other) is Complex:
            return (<Fermion>self)._div_Fermion_Complex(<Complex>other)
        if type(self) is Fermion and hasattr(other, "real") and hasattr(other, "imag"):
            return (<Fermion>self)._div_Fermion_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for Fermion.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline Fermion _div_Fermion_float(Fermion self, float other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline Fermion _div_Fermion_int(Fermion self, int other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline Fermion _div_Fermion_Complex(Fermion self, Complex other):
        cdef Fermion out = Fermion()
        out.instance[0] = self.instance[0] / other.instance
        return out


cdef class LatticeColourVector:
    cdef lattice_colour_vector.LatticeColourVector* instance
    cdef Layout layout
    cdef Py_ssize_t buffer_shape[2]
    cdef Py_ssize_t buffer_strides[2]
    cdef int view_count

    cdef lattice_colour_vector.LatticeColourVector cppobj(self):
        return self.instance[0]

    cdef validate_index(self, index):
        validate_LatticeColourVector_index(
            index, self.instance.lattice_shape(), self.instance.num_dims(),
            self.instance.volume())

    def __cinit__(self, Layout layout, *args):
        self.instance = new lattice_colour_vector.LatticeColourVector(layout.instance[0], colour_vector.ColourVector())
        self.layout = layout
        self.view_count = 0

    def __init__(self, Layout layout, *args):
        cdef int i, volume
        volume = layout.instance.volume()
        if len(args) is 1 and type(args[0]) is ColourVector:
            for i in range(volume):
                self.instance[0][i] = (<ColourVector>args[0]).instance[0]

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance.volume()
        self.buffer_strides[0] = 3 * itemsize
        self.buffer_shape[1] = 3
        self.buffer_strides[1] = itemsize

        buffer.buf = <char*>&(self.instance[0][0])
        buffer.format = "dd"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = 3 * self.instance.volume() * itemsize
        buffer.ndim = 2

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1

    def __getitem__(self, index):
        cdef int num_dims = self.instance.num_dims()
        if type(index) is tuple and len(index) == self.instance.num_dims():
            out = ColourVector()
            self.validate_index(index)
            (<ColourVector>out).instance[0] = (<LatticeColourVector>self).instance[0](<vector[unsigned int]>index)
            return out
        if type(index) is tuple and len(index) == num_dims + 1:
            out = Complex(0.0, 0.0)
            self.validate_index(index)
            validate_ColourVector_indices(index[num_dims])
            (<Complex>out).instance = (<LatticeColourVector>self).instance[0](<vector[unsigned int]>index[:num_dims])[index[num_dims]]
            return out.to_complex()
        if type(index) is int:
            out = ColourVector()
            self.validate_index(index)
            (<ColourVector>out).instance[0] = (<LatticeColourVector>self).instance[0](<int>index)
            return out
        raise TypeError("Invalid index type in LatticeColourVector.__getitem__")

    def __setitem__(self, index, value):
        cdef int num_dims = self.instance.num_dims()
        if type(value) is ColourVector:
            self.validate_index(index[:num_dims] if type(index) is tuple else index)
            self.assign_elem(index[:num_dims] if type(index) is tuple else index, (<ColourVector>value).instance[0])
            return
        elif type(value) is Complex:
            pass
        elif hasattr(value, "real") and hasattr(value, "imag") and isinstance(index, tuple):
            value = Complex(value.real, value.imag)
        else:
            value = Complex(<double?>value, 0.0)

        cdef colour_vector.ColourVector* mat
        if type(index) is tuple:
            mat = &(self.instance[0](<vector[unsigned int]?>index[:num_dims]))
        else:
            mat = &(self.instance[0](<int?>index))
        validate_ColourVector_indices(index[num_dims])
        cdef complex.Complex* z = &(mat[0][<int?>index[num_dims]])
        z[0] = (<Complex>value).instance

    cdef assign_elem(self, index, colour_vector.ColourVector value):
        cdef colour_vector.ColourVector* m
        if type(index) is tuple:
            m = &(self.instance[0](<vector[unsigned int]>index))
        else:
            m = &(self.instance[0](<int?>index))
        m[0] = value

    def adjoint(self):
        raise NotImplementedError
        out = LatticeColourVector()
        out.instance[0] = self.instance[0].adjoint()
        return out

    @staticmethod
    def zeros(Layout layout):
        return LatticeColourVector(layout, ColourVector.zeros())

    @staticmethod
    def ones(Layout layout):
        return LatticeColourVector(layout, ColourVector.ones())

    def to_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    @property
    def num_dims(self):
        return self.instance.num_dims()

    @property
    def volume(self):
        return self.instance.volume()

    @property
    def lattice_shape(self):
        return tuple(self.instance.lattice_shape())

    @property
    def shape(self):
        return tuple(self.instance.lattice_shape()) + ColourVector.shape

    def __add__(self, other):
        if type(self) is LatticeColourVector and type(other) is ColourVector:
            return (<LatticeColourVector>self)._add_LatticeColourVector_ColourVector(<ColourVector>other)
        if type(self) is LatticeColourVector and type(other) is LatticeColourVector:
            return (<LatticeColourVector>self)._add_LatticeColourVector_LatticeColourVector(<LatticeColourVector>other)
        if type(self) is LatticeColourVector and type(other) is FermionField:
            return (<LatticeColourVector>self)._add_LatticeColourVector_FermionField(<FermionField>other)
        raise TypeError("Unsupported operand types for LatticeColourVector.__add__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourVector _add_LatticeColourVector_ColourVector(LatticeColourVector self, ColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline LatticeColourVector _add_LatticeColourVector_LatticeColourVector(LatticeColourVector self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    cdef inline FermionField _add_LatticeColourVector_FermionField(LatticeColourVector self, FermionField other):
        cdef FermionField out = FermionField(self.layout)
        out.instance[0] = self.instance[0] + other.instance[0]
        return out

    def __mul__(self, other):
        if type(self) is float and type(other) is LatticeColourVector:
            return (<LatticeColourVector>other)._mul_LatticeColourVector_float(<float>self)
        if type(self) is LatticeColourVector and type(other) is float:
            return (<LatticeColourVector>self)._mul_LatticeColourVector_float(<float>other)
        if type(self) is int and type(other) is LatticeColourVector:
            return (<LatticeColourVector>other)._mul_LatticeColourVector_int(<int>self)
        if type(self) is LatticeColourVector and type(other) is int:
            return (<LatticeColourVector>self)._mul_LatticeColourVector_int(<int>other)
        if type(self) is Complex and type(other) is LatticeColourVector:
            return (<LatticeColourVector>other)._mul_LatticeColourVector_Complex(<Complex>self)
        if type(self) is LatticeColourVector and type(other) is Complex:
            return (<LatticeColourVector>self)._mul_LatticeColourVector_Complex(<Complex>other)
        if hasattr(self, "real") and hasattr(self, "imag") and type(other) is LatticeColourVector:
            return (<LatticeColourVector>other)._mul_LatticeColourVector_Complex(Complex(self.real, self.imag))
        if type(self) is LatticeColourVector and hasattr(other, "real") and hasattr(other, "imag"):
            return (<LatticeColourVector>self)._mul_LatticeColourVector_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourVector.__mul__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourVector _mul_LatticeColourVector_float(LatticeColourVector self, float other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline LatticeColourVector _mul_LatticeColourVector_int(LatticeColourVector self, int other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] * other
        return out

    cdef inline LatticeColourVector _mul_LatticeColourVector_Complex(LatticeColourVector self, Complex other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] * other.instance
        return out

    def __sub__(self, other):
        if type(self) is LatticeColourVector and type(other) is ColourVector:
            return (<LatticeColourVector>self)._sub_LatticeColourVector_ColourVector(<ColourVector>other)
        if type(self) is LatticeColourVector and type(other) is LatticeColourVector:
            return (<LatticeColourVector>self)._sub_LatticeColourVector_LatticeColourVector(<LatticeColourVector>other)
        raise TypeError("Unsupported operand types for LatticeColourVector.__sub__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourVector _sub_LatticeColourVector_ColourVector(LatticeColourVector self, ColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    cdef inline LatticeColourVector _sub_LatticeColourVector_LatticeColourVector(LatticeColourVector self, LatticeColourVector other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] - other.instance[0]
        return out

    def __div__(self, other):
        if type(self) is LatticeColourVector and type(other) is float:
            return (<LatticeColourVector>self)._div_LatticeColourVector_float(<float>other)
        if type(self) is LatticeColourVector and type(other) is int:
            return (<LatticeColourVector>self)._div_LatticeColourVector_int(<int>other)
        if type(self) is LatticeColourVector and type(other) is Complex:
            return (<LatticeColourVector>self)._div_LatticeColourVector_Complex(<Complex>other)
        if type(self) is LatticeColourVector and hasattr(other, "real") and hasattr(other, "imag"):
            return (<LatticeColourVector>self)._div_LatticeColourVector_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourVector.__div__: "
                        "{} and {}".format(type(self), type(other)))

    def __truediv__(self, other):
        if type(self) is LatticeColourVector and type(other) is float:
            return (<LatticeColourVector>self)._div_LatticeColourVector_float(<float>other)
        if type(self) is LatticeColourVector and type(other) is int:
            return (<LatticeColourVector>self)._div_LatticeColourVector_int(<int>other)
        if type(self) is LatticeColourVector and type(other) is Complex:
            return (<LatticeColourVector>self)._div_LatticeColourVector_Complex(<Complex>other)
        if type(self) is LatticeColourVector and hasattr(other, "real") and hasattr(other, "imag"):
            return (<LatticeColourVector>self)._div_LatticeColourVector_Complex(Complex(other.real, other.imag))
        raise TypeError("Unsupported operand types for LatticeColourVector.__truediv__: "
                        "{} and {}".format(type(self), type(other)))

    cdef inline LatticeColourVector _div_LatticeColourVector_float(LatticeColourVector self, float other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline LatticeColourVector _div_LatticeColourVector_int(LatticeColourVector self, int other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] / other
        return out

    cdef inline LatticeColourVector _div_LatticeColourVector_Complex(LatticeColourVector self, Complex other):
        cdef LatticeColourVector out = LatticeColourVector(self.layout)
        out.instance[0] = self.instance[0] / other.instance
        return out


cdef class FermionField:
    cdef fermion_field.FermionField* instance
    def __init__(self):
        pass


