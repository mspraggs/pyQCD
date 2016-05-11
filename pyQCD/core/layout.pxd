from libcpp.vector cimport vector

cdef extern from "layout.hpp" namespace "pyQCD":
    cdef cppclass Layout:
        Layout()
        Layout(const vector[unsigned int]&, const vector[unsigned int]&, const unsigned int, const unsigned int)
        unsigned int get_array_index(const unsigned int)
        unsigned int get_array_index(const vector[unsigned int]&)
        unsigned int get_site_index(const unsigned int)
        unsigned int num_dims()
        unsigned int local_volume()
        const vector[unsigned int]& local_shape()
