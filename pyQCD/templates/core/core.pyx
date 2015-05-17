from operators cimport *
cimport complex
{% for matrix in matrixdefs %}
cimport {{ matrix.matrix_name|to_underscores }}
cimport {{ matrix.array_name|to_underscores }}
cimport {{ matrix.lattice_matrix_name|to_underscores }}
cimport {{ matrix.lattice_array_name|to_underscores }}
{% endfor %}

cdef class Complex:
    cdef complex.Complex instance

    def __init__(self, x, y):
        cdef complex.Complex z = complex.Complex(x, y)
        self.instance = z

{% for matrix in matrixdefs %}
cdef class {{ matrix.matrix_name }}:
    cdef {{ matrix.matrix_name|to_underscores }}.{{ matrix.matrix_name }} instance

    def __getitem__(self, index):
        out = Complex()
    {% if matrix.num_cols > 1 %}
        out.instance = self.instance(index[0], index[1])
    {% else %}
        out.instance = self.instance[index]
    {% endif %}
        return out

    def __setitem__(self, index, Complex value):
    {% if matrix.num_cols > 1 %}
        cdef complex.Complex* z = &self.instance(index[0], index[1])
    {% else %}
        cdef complex.Complex* z = &self.instance[index]
    {% endif %}
        z[0] = value.instance

    def adjoint(self):
        out = {{ matrix.matrix_name }}()
        out.instance = self.instance.adjoint()
        return out

    @staticmethod
    def zeros():
        out = {{ matrix.matrix_name }}()
        out.instance = {{ matrix.matrix_name|to_underscores }}.zeros()
        return out


cdef class {{ matrix.array_name }}:
    cdef {{ matrix.array_name|to_underscores }}.{{ matrix.array_name }} instance

    def _init_with_args_(self, unsigned int N, {{ matrix.matrix_name }} value):
        self.instance = {{ matrix.array_name|to_underscores }}.{{ matrix.array_name }}(N, value.instance)

    def __init__(self, *args):
        if not args:
            pass
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], {{ matrix.matrix_name }}):
            self._init_with_args_(args[0], args[1])
        else:
            raise TypeError


cdef class {{ matrix.lattice_matrix_name }}:
    def __init__(self):
        pass


cdef class {{ matrix.lattice_array_name }}:
    def __init__(self):
        pass


{% endfor %}