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
            return "(%d+%d)" % (self.real, self.imag)

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

    def __setitem__(self, index, Complex value):
    {% if matrix.num_cols > 1 %}
        self.assign_elem(index[0], index[1], value.instance)
    {% else %}
        self.assign_elem(index, value.instance)
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
    def __{{ funcname }}__(self, other):
        {% set ops = operators[(matrix.matrix_name, funcname)] %}
        {% for ret, lhs, rhs in ops %}
        if type(self) is {{ lhs }} and type(other) is {{ rhs }}:
            out = {{ ret }}()
            {% set lhs_prefix = "<" + lhs + ">" if lhs != "float" else "<" + precision + ">" %}
            {% set rhs_prefix = "<" + rhs + ">" if rhs != "float" else "<" + precision + ">" %}
            {% set lhs_suffix = ".instance" if lhs != "float" else "" %}
            {% set rhs_suffix = ".instance" if rhs != "float" else "" %}
            {% if (lhs in ["float", "Complex"] or rhs in ["float", "Complex"]) and (matrix.num_cols == 1 or funcname == "div") %}
            out.instance = ({{ lhs_prefix }}self){{ lhs_suffix }} {{ op }} ({{ rhs_prefix }}other){{ rhs_suffix }}
            {% else %}
            out.instance = {{ lhs_prefix }}self{{ lhs_suffix }} {{ op }} {{ rhs_prefix }}other{{ rhs_suffix }}
            {% endif %}

            return out
        {% else %}
        raise NotImplementedError
        {% endfor %}

{% endfor %}

{% for ret, func, lhs, rhs in operators[matrix.matrix_name] %}
    cdef {{ func }}({{ lhs }} self, {{ rhs }} other):
        out = {{ ret }}()
        out.instance = self{% if lhs != "float" %}.instance{% endif %} * other{% if rhs != "float" %}.instance{% endif %}

        return out

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