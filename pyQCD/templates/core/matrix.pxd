from {{ typedef.element_type.cmodule }} cimport {{ typedef.element_type.cname }}
{#from {{ typedef.cmodule }} cimport {{ matrixdef.lattice_matrix_name }} #}


cdef extern from "types.hpp":
    cdef cppclass {{ typedef.cname }}:
        {{ typedef.cname }}() except +
        {{ typedef.cname }} adjoint()
        {% if typedef.shape|length > 1 %}
        {{ typedef.element_type.cname }}& operator()(int, int) except +
        {% else %}
        {{ typedef.element_type.cname }}& operator[](int) except +
        {% endif %}
        {#{{ matrixdef.lattice_matrix_name }} broadcast() except +#}


    cdef {{ typedef.cname }} zeros "{{ typedef.cname }}::Zero"()
    cdef {{ typedef.cname }} ones "{{ typedef.cname }}::Ones"()
{% if typedef.is_square %}
    cdef {{ typedef.cname }} identity "{{ typedef.cname }}::Identity"()
{% endif %}
    cdef void mat_assign({{ typedef.cname }}&, const int, const int, const Complex)
    cdef void mat_assign({{ typedef.cname }}*, const int, const int, const Complex)