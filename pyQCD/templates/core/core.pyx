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
    cdef {{matrix.matrix_name|to_underscores }}.{{ matrix.matrix_name }} instance

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

cdef class {{ matrix.array_name }}:
    def __init__(self):
        pass

cdef class {{ matrix.lattice_matrix_name }}:
    def __init__(self):
        pass

cdef class {{ matrix.lattice_array_name }}:
    def __init__(self):
        pass

{% endfor %}