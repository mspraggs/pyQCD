cdef extern from "core/types.hpp" namespace "pyQCD::python":
    cdef cppclass _{{ typedef.cname }} "pyQCD::python::{{ typedef.cname }}":
        _{{ typedef.cname }}() except +
        _{{ typedef.cname }}(const _{{ typedef.cname }}&) except +
        _{{ typedef.cname }} adjoint()
        {% if typedef.ndims > 1 %}
        {{ typedef.element_type.cname }}& operator()(int, int) except +
        {% else %}
        {{ typedef.element_type.cname }}& operator[](int) except +
        {% endif %}


    cdef _{{ typedef.cname }} _{{ typedef.cname }}_zeros "pyQCD::python::{{ typedef.cname }}::Zero"()
    cdef _{{ typedef.cname }} _{{ typedef.cname }}_ones "pyQCD::python::{{ typedef.cname }}::Ones"()

cdef class {{ typedef.cname }}:
    cdef _{{ typedef.cname }}* instance
    cdef int view_count
    cdef Py_ssize_t buffer_shape[{{ typedef.ndims }}]
    cdef Py_ssize_t buffer_strides[{{ typedef.ndims }}]
