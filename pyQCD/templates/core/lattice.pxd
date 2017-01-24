cdef extern from "core/types.hpp" namespace "pyQCD::python":
    cdef cppclass _{{ typedef.cname }} "pyQCD::python::{{ typedef.cname }}":
        _{{ typedef.cname }}() except +
        _{{ typedef.cname }}(const Layout&, const _{{ typedef.element_type.cname }}&, unsigned int site_size) except +
        _{{ typedef.element_type.cname }}& operator[](const unsigned int)
        unsigned int volume()
        unsigned int num_dims()
        const vector[unsigned int]& lattice_shape()

cdef class {{ typedef.cname }}:
    cdef _{{ typedef.cname }}* instance
    cdef Layout* lexico_layout
    cdef int view_count
    cdef int site_size
    cdef Py_ssize_t buffer_shape[{{ typedef.buffer_ndims }}]
    cdef Py_ssize_t buffer_strides[{{ typedef.buffer_ndims }}]