cdef extern from "core/types.hpp" namespace "pyQCD::python":
    cdef cppclass _{{ typedef.cname }} "pyQCD::python::{{ typedef.cname }}":
        _{{ typedef.cname }}() except +
        _{{ typedef.cname }}(const _Layout&, const _{{ typedef.element_type.cname }}&, unsigned int site_size) except +
        _{{ typedef.element_type.cname }}& operator[](const unsigned int)
        unsigned int volume()
        unsigned int num_dims()
        const vector[unsigned int]& lattice_shape()

cdef class {{ typedef.cname }}:
    cdef _{{ typedef.cname }}* instance
    cdef public Layout layout
    cdef bool_t is_buffer_compatible
    cdef int view_count
    cdef public int site_size
    cdef Py_ssize_t buffer_shape[{{ typedef.element_type.ndims + 1 }}]
    cdef Py_ssize_t buffer_strides[{{ typedef.element_type.ndims + 1 }}]