from cpython cimport Py_buffer
from libcpp.vector cimport vector

import numpy as np

cimport complex
cimport layout
from operators cimport *
{% for matrix in matrixdefs %}
cimport {{ matrix.matrix_name|to_underscores }}
cimport {{ matrix.array_name|to_underscores }}
cimport {{ matrix.lattice_matrix_name|to_underscores }}
cimport {{ matrix.lattice_array_name|to_underscores }}
{% endfor %}

{% import "core/arithmetic.pyx" as arithmetic %}

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


{% for num_rows, num_cols, matrix_name, array_name, lattice_matrix_name, lattice_array_name in matrixdefs %}
{% set is_matrix = num_cols > 1 %}
{% set is_square = num_cols == num_rows %}
{% set cmatrix = matrix_name|to_underscores + "." + matrix_name %}
{% set carray = array_name|to_underscores + "." + array_name %}
{% set clattice_matrix = lattice_matrix_name|to_underscores + "." + lattice_matrix_name %}
{% set clattice_array = lattice_array_name|to_underscores + "." + lattice_array_name %}
cdef inline int validate_{{ matrix_name }}_indices(int i{% if is_matrix %}, int j{% endif %}) except -1:
    if i > {{ num_rows - 1}} or i < 0{% if is_matrix %} or j > {{ num_cols - 1 }} or j < 0{% endif %}:
        raise IndexError("Indices in {{ matrix_name }} element access out of bounds: "
                         "{}".format((i{% if is_matrix %}, j{% endif %})))


cdef class {{ matrix_name }}:
    cdef {{ cmatrix }}* instance
    cdef Py_ssize_t buffer_shape[{% if is_matrix %}2{% else %}1{% endif %}]
    cdef Py_ssize_t buffer_strides[{% if is_matrix %}2{% else %}1{% endif %}]
    shape = ({{ num_rows}},{% if is_matrix %} {{num_cols}}{% endif %})

    cdef {{ cmatrix }} cppobj(self):
        return self.instance[0]

    cdef validate_indices(self, int i{% if is_matrix %}, int j {% endif %}):
        validate_{{ matrix_name }}_indices(i{% if is_matrix %}, j{% endif %})

    def __cinit__(self):
        self.instance = new {{ cmatrix }}()

    def __init__(self, *args):
        cdef int i, j
        if not args:
            pass
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            for i, elem in enumerate(args[0]):
{% if is_matrix %}
                for j, subelem in enumerate(elem):
                    self.validate_indices(i, j)
                    self[i, j] = subelem
{% else %}
                self.validate_indices(i)
                self[i] = elem
{% endif %}

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = {{ num_rows }}
        self.buffer_strides[0] = itemsize
        {% if is_matrix %}
        self.buffer_shape[1] = {{ num_cols }}
        self.buffer_strides[1] = {{ num_cols }} * itemsize
        {% endif %}

        buffer.buf = <char*>self.instance
        {% set num_format = "d" if precision == "double" else "f" %}
        buffer.format = "{{ num_format + num_format }}"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = {{ num_rows }} * {{ num_cols }} * itemsize
        buffer.ndim = {% if is_matrix %}2{% else %}1{% endif %}

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
{% if is_matrix %}
            self.validate_indices(index[0], index[1])
            out.instance = self.instance[0](index[0], index[1])
{% else %}
            self.validate_indices(index[0])
            out.instance = self.instance[0][index[0]]
        elif type(index) is int:
            self.validate_indices(index)
            out.instance = self.instance[0][index]
{% endif %}
        else:
            raise TypeError("Invalid index type in {{ matrix_name }}.__setitem__: "
                            "{}".format(type(index)))
        return out.to_complex()

    def __setitem__(self, index, value):
        if type(value) is Complex:
            pass
        elif hasattr(value, 'real') and hasattr(value, 'imag'):
            value = Complex(<{{ precision }}?>(value.real),
                            <{{ precision }}?>(value.imag))
        else:
            value = Complex(<{{ precision }}?>value, 0.0)
        if type(index) is tuple:
{% if is_matrix %}
            self.validate_indices(index[0], index[1])
            self.assign_elem(index[0], index[1], (<Complex>value).instance)
{% else %}
            self.validate_indices(index[0])
            self.assign_elem(index[0], (<Complex>value).instance)
        elif type(index) is int:
            self.validate_indices(index)
            self.assign_elem(index, (<Complex>value).instance)
{% endif %}
        else:
            raise TypeError("Invalid index type in {{ matrix_name }}.__setitem__: "
                            "{}".format(type(index)))

{% if is_matrix %}
    cdef void assign_elem(self, int i, int j, complex.Complex value):
        {{ matrix_name|to_underscores }}.mat_assign(self.instance, i, j, value)
{% else %}
    cdef void assign_elem(self, int i, complex.Complex value):
        cdef complex.Complex* z = &(self.instance[0][i])
        z[0] = value
{% endif %}

    def adjoint(self):
        out = {{ matrix_name }}()
        out.instance[0] = self.instance[0].adjoint()
        return out

    @staticmethod
    def zeros():
        out = {{ matrix_name }}()
        out.instance[0] = {{ matrix_name|to_underscores }}.zeros()
        return out

    @staticmethod
    def ones():
        out = {{ matrix_name }}()
        out.instance[0] = {{ matrix_name|to_underscores }}.ones()
        return out

{% if is_square %}
    @staticmethod
    def identity():
        out = {{ matrix_name }}()
        out.instance[0] = {{ matrix_name|to_underscores }}.identity()
        return out

{% endif %}
    def to_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

{{ arithmetic.arithmetic_ops(operators, matrix_name, scalar_types) }}

cdef class {{ array_name }}:
    cdef {{ carray }}* instance
    cdef Py_ssize_t buffer_shape[{% if is_matrix %}3{% else %}2{% endif %}]
    cdef Py_ssize_t buffer_strides[{% if is_matrix %}3{% else %}2{% endif %}]
    cdef int view_count

    cdef {{ carray }} cppobj(self):
        return self.instance[0]

    cdef _init_with_args_(self, unsigned int N, {{ matrix_name }} value):
        self.instance[0] = {{ array_name|to_underscores }}.{{ array_name }}(N, value.instance[0])

    cdef validate_index(self, int i):
        if i >= self.instance.size() or i < 0:
            raise IndexError("Index in {{ array_name }} element access out of bounds: "
                             "{}".format(i))

    def __cinit__(self):
        self.instance = new {{ carray }}()
        self.view_count = 0

    def __init__(self, *args):
        cdef int i, N
        if not args:
            pass
        elif len(args) == 1 and hasattr(args[0], "__len__"):
            N = len(args[0])
            self.instance.resize(N)
            for i in range(N):
                self[i] = {{ matrix_name }}(args[0][i])
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], {{ matrix_name }}):
            self._init_with_args_(args[0], args[1])
        else:
            raise TypeError("{{ array_name }} constructor expects "
                            "either zero or two arguments")

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

        self.buffer_shape[0] = self.instance.size()
        self.buffer_strides[0] = itemsize * {{ num_rows * num_cols }}
        self.buffer_shape[1] = {{ num_rows }}
        self.buffer_strides[1] = itemsize
        {% if is_matrix %}
        self.buffer_shape[2] = {{ num_cols }}
        self.buffer_strides[2] = {{ num_cols }} * itemsize
        {% endif %}

        buffer.buf = <char*>&(self.instance[0][0])
        {% set num_format = "d" if precision == "double" else "f" %}
        buffer.format = "{{ num_format + num_format }}"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = {{ num_rows }} * {{ num_cols }} * itemsize
        buffer.ndim = {% if is_matrix %}3{% else %}2{% endif %}

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
            out = {{ matrix_name }}()
            (<{{ matrix_name }}>out).instance[0] = (self.instance[0])[<int?>(index[0])]
            return out
        elif type(index) is tuple:
            self.validate_index(index[0])
            out = Complex(0.0, 0.0)
{% if is_matrix %}
            validate_{{ matrix_name }}_indices(index[1], index[2])
            (<Complex>out).instance = self.instance[0][<int?>index[0]](<int?>index[1], <int?>index[2])
{% else %}
            validate_{{ matrix_name }}_indices(index[1])
            (<Complex>out).instance = self.instance[0][<int?>index[0]][<int?>index[1]]
{% endif %}
            return out.to_complex()
        else:
            self.validate_index(index)
            out = {{ matrix_name }}()
            (<{{ matrix_name }}>out).instance[0] = self.instance[0][<int?>index]
            return out

    def __setitem__(self, index, value):
        if type(value) is {{ matrix_name }}:
            self.validate_index(index[0] if type(index) is tuple else index)
            self.assign_elem(index[0] if type(index) is tuple else index, (<{{ matrix_name }}>value).instance[0])
            return
        elif type(value) is Complex:
            pass
        elif hasattr(value, "real") and hasattr(value, "imag") and isinstance(index, tuple):
            value = Complex(value.real, value.imag)
        else:
            value = Complex(<{{ precision }}?>value, 0.0)

        cdef {{ cmatrix }}* mat = &(self.instance[0][<int?>index[0]])
{% if is_matrix %}
        validate_{{ matrix_name }}_indices(index[1], index[2])
        {{ matrix_name|to_underscores }}.mat_assign(mat, <int?>index[1], <int?>index[2], (<Complex?>value).instance)
{% else %}
        validate_{{ matrix_name }}_indices(index[1])
        cdef complex.Complex* z = &(mat[0][<int?>index[1]])
        z[0] = (<Complex>value).instance
{% endif %}

    cdef void assign_elem(self, int i, {{ cmatrix }} value):
        cdef {{ cmatrix }}* m = &(self.instance[0][i])
        m[0] = value

    def adjoint(self):
        out = {{ array_name }}()
        out.instance[0] = self.instance[0].adjoint()
        return out

    @staticmethod
    def zeros(int num_elements):
        out = {{ array_name }}()
        out.instance[0] = {{ carray }}(num_elements, {{ matrix_name|to_underscores }}.zeros())
        return out

    @staticmethod
    def ones(int num_elements):
        out = {{ array_name }}()
        out.instance[0] = {{ carray }}(num_elements, {{ matrix_name|to_underscores }}.ones())
        return out

{% if is_square %}
    @staticmethod
    def identity(int num_elements):
        out = {{ array_name }}()
        out.instance[0] = {{ carray }}(num_elements, {{ matrix_name|to_underscores }}.identity())
        return out

{% endif %}
    def to_numpy(self):
        out = np.asarray(self)
        out.dtype = complex
        return out

    @property
    def size(self):
        return self.instance.size()

    @property
    def shape(self):
        return (self.size, {{ num_rows}},{% if is_matrix %} {{num_cols}}{% endif %})

{{ arithmetic.arithmetic_ops(operators, array_name, scalar_types) }}

cdef class {{ lattice_matrix_name }}:
    cdef {{ clattice_matrix }}* instance

    cdef {{ clattice_matrix }} cppobj(self):
        return self.instance[0]

    cdef validate_index(self, index):
        cdef int i
        cdef int num_dims = self.instance.num_dims()
        cdef vector[unsigned int] shape = self.instance.lattice_shape()
        if type(index) is tuple:
            for i in range(num_dims):
                if index[i] >= shape[i] or index[i] < 0:
                    raise IndexError("Index in {{ lattice_matrix_name }} element access "
                                     "out of bounds: {}".format(index))
        elif type(index) is int:
            if index < 0 or index >= self.instance.volume():
                raise IndexError("Index in {{ lattice_matrix_name }} element access "
                                 "out of bounds: {}".format(index))

    def __cinit__(self, Layout layout, *args):
        self.instance = new {{ clattice_matrix }}(layout.instance[0], {{ cmatrix }}())

    def __init__(self, Layout layout, *args):
        cdef int i, volume
        volume = layout.instance.volume()
        if len(args) is 1 and type(args[0]) is {{ matrix_name }}:
            for i in range(volume):
                self.instance[0][i] = (<{{ matrix_name }}>args[0]).instance[0]

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        pass

    def __releasebuffer__(self, Py_buffer* buffer):
        pass

    def __getitem__(self, index):
        cdef int num_dims = self.instance.num_dims()
        if type(index) is tuple and len(index) == self.instance.num_dims():
            out = {{ matrix_name }}()
            self.validate_index(index)
            (<{{ matrix_name }}>out).instance[0] = (<{{ lattice_matrix_name }}>self).instance[0](<vector[unsigned int]>index)
            return out
        if type(index) is tuple and len(index) == num_dims + {% if is_matrix %}2{% else %}1{% endif %}:
            out = Complex(0.0, 0.0)
            self.validate_index(index)
            validate_{{ matrix_name }}_indices(index[num_dims]{% if is_matrix %}, index[num_dims + 1]{% endif %})
{% if is_matrix %}
            (<Complex>out).instance = (<{{ lattice_matrix_name }}>self).instance[0](<vector[unsigned int]>index[:num_dims])(index[num_dims], index[num_dims + 1])
{% else %}
            (<Complex>out).instance = (<{{ lattice_matrix_name }}>self).instance[0](<vector[unsigned int]>index[:num_dims])[index[num_dims]]
{% endif %}
            return out.to_complex()
        if type(index) is int:
            out = {{ matrix_name }}()
            self.validate_index(index)
            (<{{ matrix_name }}>out).instance[0] = (<{{ lattice_matrix_name }}>self).instance[0](<int>index)
            return out
        raise TypeError("Invalid index type in {{ lattice_matrix_name }}.__getitem__")

    def __setitem__(self, index, value):
        cdef int num_dims = self.instance.num_dims()
        if type(value) is {{ matrix_name }}:
            self.validate_index(index[:num_dims] if type(index) is tuple else index)
            self.assign_elem(index[:num_dims] if type(index) is tuple else index, (<{{ matrix_name }}>value).instance[0])
            return
        elif type(value) is Complex:
            pass
        elif hasattr(value, "real") and hasattr(value, "imag") and isinstance(index, tuple):
            value = Complex(value.real, value.imag)
        else:
            value = Complex(<{{ precision }}?>value, 0.0)

        cdef {{ cmatrix }}* mat
        if type(index) is tuple:
            mat = &(self.instance[0](<vector[unsigned int]?>index[:num_dims]))
        else:
            mat = &(self.instance[0](<int?>index))
{% if is_matrix %}
        validate_{{ matrix_name }}_indices(index[num_dims], index[num_dims + 1])
        {{ matrix_name|to_underscores }}.mat_assign(mat, <int?>index[num_dims], <int?>index[num_dims + 1], (<Complex?>value).instance)
{% else %}
        validate_{{ matrix_name }}_indices(index[num_dims])
        cdef complex.Complex* z = &(mat[0][<int?>index[num_dims]])
        z[0] = (<Complex>value).instance
{% endif %}

    cdef assign_elem(self, index, {{ cmatrix }} value):
        cdef {{ cmatrix }}* m
        if type(index) is tuple:
            m = &(self.instance[0](<vector[unsigned int]>index))
        else:
            m = &(self.instance[0](<int?>index))
        m[0] = value

    def adjoint(self):
        pass

    @staticmethod
    def zeros():
        pass

    @staticmethod
    def ones():
        pass

{% if is_square %}
    @staticmethod
    def identity():
        pass
    
{% endif %}
    def to_numpy(self):
        pass

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
        return tuple(self.instance.lattice_shape()) + {{ matrix_name }}.shape

{{ arithmetic.arithmetic_ops(operators, lattice_matrix_name, scalar_types) }}


cdef class {{ lattice_array_name }}:
    cdef {{ clattice_array }}* instance
    def __init__(self):
        pass


{% endfor %}