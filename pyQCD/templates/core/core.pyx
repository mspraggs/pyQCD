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

    @property
    def real(self):
        return self.instance.real()

    @property
    def imag(self):
        return self.instance.imag()

    def __repr__(self):
        if self.real == 0:
            return "%dj" % self.imag
        else:
            return "(%d%s%dj)" % (self.real, ('+' if self.imag >= 0 else ''),
                                  self.imag)

{% for matrix in matrixdefs %}
{% set cmatrix = matrix.matrix_name|to_underscores + "." + matrix.matrix_name %}
{% set carray = matrix.array_name|to_underscores + "." + matrix.array_name %}
{% set clattice_matrix = matrix.lattice_matrix_name|to_underscores + "." + matrix.lattice_matrix_name %}
{% set clattice_array = matrix.lattice_array_name|to_underscores + "." + matrix.lattice_array_name %}
cdef class {{ matrix.matrix_name }}:
    cdef {{ cmatrix }} instance

    def __getitem__(self, index):
        out = Complex(0.0, 0.0)
    {% if matrix.num_cols > 1 %}
        out.instance = self.instance(index[0], index[1])
    {% else %}
        out.instance = self.instance[index]
    {% endif %}
        return out

    def __setitem__(self, index, value):
        if type(value) == Complex:
            pass
        elif type(value) == complex:
            value = Complex(value.real, value.imag)
        elif type(value) == float:
            value = Complex(value, 0.0)
        else:
            raise TypeError("Invalid value type in {{ matrix.matrix_name }}.__setitem__: "
                            "{}".format(type(value)))
    {% if matrix.num_cols > 1 %}
        self.assign_elem(index[0], index[1], (<Complex>value).instance)
    {% else %}
        self.assign_elem(index, (<Complex>value).instance)
    {% endif %}

    {% if matrix.num_cols > 1 %}
    cdef void assign_elem(self, int i, int j, complex.Complex value):
        {{ matrix.matrix_name|to_underscores }}.mat_assign(self.instance, i, j, value)
    {% else %}
    cdef void assign_elem(self, int i, complex.Complex value):
        cdef complex.Complex* z = &(self.instance[i])
        z[0] = value
    {% endif %}

    def adjoint(self):
        out = {{ matrix.matrix_name }}()
        out.instance = self.instance.adjoint()
        return out

    @staticmethod
    def zeros():
        out = {{ matrix.matrix_name }}()
        out.instance = {{ matrix.matrix_name|to_underscores }}.zeros()
        return out

{% for funcname, op in zip(["add", "sub", "mul", "div"], "+-*/") %}
{% set ops = operators[(matrix.matrix_name, funcname)] %}
    def __{{ funcname }}__(self, other):
{% for ret, lhs, rhs in ops %}
        if type(self) is {{ lhs }} and type(other) is {{ rhs }}:
{% if lhs == "float" or lhs == "Complex" %}
            return other._{{ funcname }}_{{ rhs }}_{{ lhs }}(<{{ lhs }}>self)
{% else %}
            return self._{{ funcname }}_{{ lhs }}_{{ rhs }}(<{{ rhs }}>other)
{% endif %}
{% endfor %}
        raise TypeError("Unsupported operand types for {{ matrix.matrix_name }}.__{{ funcname }}__: "
                        "{} and {}".format(type(self), type(other)))

{% for ret, lhs, rhs in ops %}
{% if lhs != "float" and lhs != "Complex" %}
    cpdef {{ ret }} _{{ funcname }}_{{ lhs }}_{{ rhs }}({{ lhs }} self, {{ rhs }} other):
        out = {{ ret }}()
        out.instance = self.instance {{ op }} other{% if rhs != "float" %}.instance{% endif %}

        return out

{% endif %}
{% endfor %}

{% endfor %}

cdef class {{ matrix.array_name }}:
    cdef {{ carray }} instance

    def _init_with_args_(self, unsigned int N, {{ matrix.matrix_name }} value):
        self.instance = {{ matrix.array_name|to_underscores }}.{{ matrix.array_name }}(N, value.instance)

    def __init__(self, *args):
        if not args:
            pass
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], {{ matrix.matrix_name }}):
            self._init_with_args_(args[0], args[1])
        else:
            raise TypeError("{{ matrix.array_name }} constructor expects "
                            "either zero or two arguments")

    def __getitem__(self, index):
        out = {{ matrix.matrix_name }}()
        out.instance = self.instance[index]
        return out

    def __setitem__(self, index, {{ matrix.matrix_name }} value):
        self.assign_elem(index, value.instance)

    cdef void assign_elem(self, int i, {{ cmatrix }} value):
        cdef {{ cmatrix }}* m = &(self.instance[i])
        m[0] = value

    def adjoint(self):
        out = {{ matrix.array_name }}()
        out.instance = self.instance.adjoint()
        return out

    @staticmethod
    def zeros(int num_elements):
        cdef {{ cmatrix }} mat = {{ matrix.matrix_name|to_underscores }}.zeros()
        out = {{ matrix.array_name }}()
        out.instance = {{ carray }}(num_elements, mat)
        return out


cdef class {{ matrix.lattice_matrix_name }}:
    cdef {{ clattice_matrix }} instance
    def __init__(self):
        pass


cdef class {{ matrix.lattice_array_name }}:
    cdef {{ clattice_array }} instance
    def __init__(self):
        pass


{% endfor %}