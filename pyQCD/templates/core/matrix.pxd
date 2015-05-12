cdef extern from "types.hpp":
    cdef cppclass {{ matrixdef.matrix_name }}:
        {{ matrixdef.matrix_name }}() except +
        {{ matrixdef.matrix_name }} adjoint()
        {% if matrixdef.num_cols > 1 %}
        const Complex& operator()(int, int) except +
        {% else %}
        const Complex& operator[](int) except +
        {% endif %}