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

        if type(site_label) == int:
            return self.instance.get_array_index(<unsigned int>site_label)
        if type(site_label) == list or type(site_label) == tuple:
            return self.instance.get_array_index(<vector[unsigned int]>site_label)
        raise TypeError("Unknown type in Layout.get_array_index: {}"
                        .format(type(site_label)))

    def get_site_index(self, unsigned int array_index):
        return self.instance.get_site_index(array_index)

    def num_dims(self):
        return self.instance.num_dims()

    def volume(self):
        return self.instance.volume()


cdef class LexicoLayout(Layout):

    def __init__(self, shape):
        self.instance = <layout.Layout*>new layout.LexicoLayout(<vector[unsigned int]?>shape)

    def __dealloc__(self):
        del self.instance


{% for matrix in matrixdefs %}
{% set is_matrix = matrix.num_cols > 1 %}
{% set is_square = matrix.num_cols == matrix.num_rows %}
{% set cmatrix = matrix.matrix_name|to_underscores + "." + matrix.matrix_name %}
{% set carray = matrix.array_name|to_underscores + "." + matrix.array_name %}
{% set clattice_matrix = matrix.lattice_matrix_name|to_underscores + "." + matrix.lattice_matrix_name %}
{% set clattice_array = matrix.lattice_array_name|to_underscores + "." + matrix.lattice_array_name %}
cdef class {{ matrix.matrix_name }}:
    cdef {{ cmatrix }}* instance

    cdef {{ cmatrix }} cppobj(self):
        return self.instance[0]

    cdef validate_indices(self, int i{% if is_matrix %}, int j {% endif %}):
        if i > {{ matrix.num_rows - 1}}{% if is_matrix %} or j > {{ matrix.num_cols - 1 }}{% endif %}:
            raise IndexError("Indices in {{ matrix.matrix_name }} element access out of bounds: "
                             "{}".format((i{% if is_matrix %}, j{% endif %})))

    def __cinit__(self):
        self.instance = new {{ cmatrix }}()

    def __init__(self, *args):
        cdef int i, j
        if not args:
            pass
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            for i, elem in enumerate(args[0]):
                if i > {{ matrix.num_rows - 1 }}:
                    raise ValueError("{{ matrix.matrix_name }}.__init__: "
                                     "First dimension of iterable > {{ matrix.num_rows }}")
{% if is_matrix %}
                for j, subelem in enumerate(elem):
                    if j > {{ matrix.num_cols - 1 }}:
                        raise ValueError("{{ matrix.matrix_name }}.__init__: "
                                         "Second dimension of iterable > {{ matrix.num_cols }}")
                    self[i, j] = subelem
{% else %}
                self[i] = elem
{% endif %}

    def __dealloc__(self):
        del self.instance

    def __getitem__(self, index):
        out = Complex(0.0, 0.0)
        if type(index) == tuple:
{% if is_matrix %}
            self.validate_indices(index[0], index[1])
            out.instance = self.instance[0](index[0], index[1])
{% else %}
            self.validate_indices(index[0])
            out.instance = self.instance[0][index[0]]
        elif type(index) == int:
            self.validate_indices(index)
            out.instance = self.instance[0][index]
{% endif %}
        else:
            raise TypeError("Invalid index type in {{ matrix.matrix_name }}.__setitem__: "
                            "{}".format(type(index)))
        return out.to_complex()

    def __setitem__(self, index, value):
        if type(value) == Complex:
            pass
        elif hasattr(value, 'real') and hasattr(value, 'imag'):
            value = Complex(<{{ precision }}?>(value.real),
                            <{{ precision }}?>(value.imag))
        else:
            value = Complex(<{{ precision }}?>value, 0.0)
        if type(index) == tuple:
{% if is_matrix %}
            self.validate_indices(index[0], index[1])
            self.assign_elem(index[0], index[1], (<Complex>value).instance)
{% else %}
            self.validate_indices(index[0])
            self.assign_elem(index[0], (<Complex>value).instance)
        elif type(index) == int:
            self.validate_indices(index)
            self.assign_elem(index, (<Complex>value).instance)
{% endif %}
        else:
            raise TypeError("Invalid index type in {{ matrix.matrix_name }}.__setitem__: "
                            "{}".format(type(index)))

{% if is_matrix %}
    cdef void assign_elem(self, int i, int j, complex.Complex value):
        {{ matrix.matrix_name|to_underscores }}.mat_assign(self.instance, i, j, value)
{% else %}
    cdef void assign_elem(self, int i, complex.Complex value):
        cdef complex.Complex* z = &(self.instance[0][i])
        z[0] = value
{% endif %}

    def adjoint(self):
        out = {{ matrix.matrix_name }}()
        out.instance[0] = self.instance[0].adjoint()
        return out

    @staticmethod
    def zeros():
        out = {{ matrix.matrix_name }}()
        out.instance[0] = {{ matrix.matrix_name|to_underscores }}.zeros()
        return out

    @staticmethod
    def ones():
        out = {{ matrix.matrix_name }}()
        out.instance[0] = {{ matrix.matrix_name|to_underscores }}.ones()
        return out

{% if is_square %}
    @staticmethod
    def identity():
        out = {{ matrix.matrix_name }}()
        out.instance[0] = {{ matrix.matrix_name|to_underscores }}.identity()
        return out

{% endif %}
    def to_numpy(self):
        out = np.zeros(self.shape, dtype=np.complex)
        for index in np.ndindex(self.shape):
            out[index] = self[index]
        return out

    @property
    def shape(self):
        return ({{ matrix.num_rows}},{% if is_matrix %} {{matrix.num_cols}}{% endif %})

{{ arithmetic.arithmetic_ops(operators, matrix, matrix.matrix_name) }}

cdef class {{ matrix.array_name }}:
    cdef {{ carray }}* instance

    cdef {{ carray }} cppobj(self):
        return self.instance[0]

    cdef _init_with_args_(self, unsigned int N, {{ matrix.matrix_name }} value):
        self.instance[0] = {{ matrix.array_name|to_underscores }}.{{ matrix.array_name }}(N, value.instance[0])

    def __cinit__(self):
        self.instance = new {{ carray }}()

    def __init__(self, *args):
        cdef int i, N
        if not args:
            pass
        elif len(args) == 1 and hasattr(args[0], "__len__"):
            N = len(args[0])
            self.instance.resize(N)
            for i in range(N):
                self[i] = {{ matrix.matrix_name }}(args[0][i])
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], {{ matrix.matrix_name }}):
            self._init_with_args_(args[0], args[1])
        else:
            raise TypeError("{{ matrix.array_name }} constructor expects "
                            "either zero or two arguments")

    def __dealloc__(self):
        del self.instance

    def __getitem__(self, index):
        out = {{ matrix.matrix_name }}()
        out.instance[0] = self.instance[0][index]
        return out

    def __setitem__(self, index, value):
        if type(value) == {{ matrix.matrix_name }}:
            self.assign_elem(index, (<{{ matrix.matrix_name }}>value).instance[0])
            return
        elif type(value) == Complex:
            pass
        elif hasattr(value, "real") and hasattr(value, "imag") and isinstance(index, tuple):
            value = Complex(value.real, value.imag)
        else:
            value = Complex(<{{ precision }}?>value, 0.0)

        cdef {{ cmatrix }}* mat = &(self.instance[0][<int?>index[0]])
{% if is_matrix %}
        {{ matrix.matrix_name|to_underscores }}.mat_assign(mat, <int?>index[1], <int?>index[2], (<Complex?>value).instance)
{% else %}
        cdef complex.Complex* z = &(mat[0][<int?>index[1]])
        z[0] = (<Complex>value).instance
{% endif %}

    cdef void assign_elem(self, int i, {{ cmatrix }} value):
        cdef {{ cmatrix }}* m = &(self.instance[0][i])
        m[0] = value

    def adjoint(self):
        out = {{ matrix.array_name }}()
        out.instance[0] = self.instance[0].adjoint()
        return out

    @staticmethod
    def zeros(int num_elements):
        out = {{ matrix.array_name }}()
        out.instance[0] = {{ carray }}(num_elements, {{ matrix.matrix_name|to_underscores }}.zeros())
        return out

    @staticmethod
    def ones(int num_elements):
        out = {{ matrix.array_name }}()
        out.instance[0] = {{ carray }}(num_elements, {{ matrix.matrix_name|to_underscores }}.ones())
        return out

{% if is_square %}
    @staticmethod
    def identity(int num_elements):
        out = {{ matrix.array_name }}()
        out.instance[0] = {{ carray }}(num_elements, {{ matrix.matrix_name|to_underscores }}.identity())
        return out

{% endif %}
    def to_numpy(self):
        cdef int i
        out = np.zeros(self.shape, dtype=np.complex)
        for i in range(self.size):
            out[i] = self[i].to_numpy()
        return out

    @property
    def size(self):
        return self.instance.size()

    @property
    def shape(self):
        return (self.size, {{ matrix.num_rows}},{% if is_matrix %} {{matrix.num_cols}}{% endif %})

{{ arithmetic.arithmetic_ops(operators, matrix, matrix.array_name) }}

cdef class {{ matrix.lattice_matrix_name }}:
    cdef {{ clattice_matrix }}* instance
    def __init__(self):
        pass


cdef class {{ matrix.lattice_array_name }}:
    cdef {{ clattice_array }}* instance
    def __init__(self):
        pass


{% endfor %}