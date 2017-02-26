{% include "edit_warning.txt" %}

from cpython cimport Py_buffer

from libcpp cimport bool as bool_t
from libcpp.vector cimport vector

from atomics cimport Real, Complex

cdef extern from "core/layout.hpp" namespace "pyQCD":
    cdef cppclass _Layout "pyQCD::Layout":
        _Layout(const vector[unsigned int]&) except+
        unsigned int get_array_index(const unsigned int)
        unsigned int get_array_index(const vector[unsigned int]&)
        unsigned int get_site_index(const unsigned int)
        unsigned int num_dims()
        unsigned int volume()
        const vector[unsigned int]& shape()

    cdef cppclass _LexicoLayout "pyQCD::LexicoLayout"(_Layout):
        _LexicoLayout(const vector[unsigned int]&) except+

    cdef cppclass _EvenOddLayout "pyQCD::EvenOddLayout"(_Layout):
        _EvenOddLayout(const vector[unsigned int]&) except+


cdef class Layout:
    cdef _Layout* instance


cdef class LexicoLayout(Layout):
    pass


cdef class EvenOddLayout(Layout):
    pass


{% for typedef in typedefs %}
    {% with typedef = typedef %}

        {% include typedef.def_template %}

    {% endwith %}
{% endfor %}