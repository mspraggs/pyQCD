from complex cimport Complex
from {{ matrixdef.lattice_matrix_name|to_underscores }} cimport {{ matrixdef.lattice_matrix_name }}


cdef extern from "types.hpp":
    cdef cppclass {{ matrixdef.matrix_name }}:
        {{ matrixdef.matrix_name }}() except +
        {{ matrixdef.matrix_name }} adjoint()
        {% if matrixdef.num_cols > 1 %}
        Complex& operator()(int, int) except +
        {% else %}
        Complex& operator[](int) except +
        {% endif %}
        {{ matrixdef.lattice_matrix_name }} broadcast() except +


    cdef {{ matrixdef.matrix_name }} zeros "{{ matrixdef.matrix_name }}::Zero"()
    cdef {{ matrixdef.matrix_name }} ones "{{ matrixdef.matrix_name }}::Ones"()
{% if matrixdef.num_cols == matrixdef.num_rows %}
    cdef {{ matrixdef.matrix_name }} identity "{{ matrixdef.matrix_name }}::Identity"()
{% endif %}
    cdef void mat_assign({{ matrixdef.matrix_name }}&, const int, const int, const Complex)