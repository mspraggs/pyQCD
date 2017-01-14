from libcpp.vector cimport vector

cdef extern from "core/layout.hpp" namespace "pyQCD":
    cdef cppclass Layout:
        Layout()
        unsigned int get_array_index(const unsigned int)
        unsigned int get_array_index(const vector[unsigned int]&)
        unsigned int get_site_index(const unsigned int)
        unsigned int num_dims()
        unsigned int volume()
        const vector[unsigned int]& shape()

    cdef cppclass LexicoLayout(Layout):
        LexicoLayout() except+
        LexicoLayout(const vector[unsigned int]&) except+