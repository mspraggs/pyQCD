from complex cimport Complex


cdef extern from "types.hpp":
    cdef cppclass {{ matrixdef.matrix_name }}:
        {{ matrixdef.matrix_name }}() except +
        {{ matrixdef.matrix_name }} adjoint()
        {% if matrixdef.num_cols > 1 %}
        Complex& operator()(int, int) except +
        {% else %}
        Complex& operator[](int) except +
        {% endif %}

    cdef {{ matrixdef.matrix_name }} zeros "{{ matrixdef.matrix_name }}::Zero"()