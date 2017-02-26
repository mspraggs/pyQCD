{% include "edit_warning.txt" %}

from cpython cimport Py_buffer
from libcpp.vector cimport vector

import numpy as np

cimport atomics
cimport core
from core cimport {% for td in typedefs %}_{{ td.name }}, {{ td.name }}{% if not loop.last %}, {% endif %}{% endfor %}

cdef class Layout:

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This class is pure-virtual. Please use a "
                                  "layout class that inherits from it.")

    property shape:
        def __get__(self):
            """The layout shape"""
            return self.instance.shape()

    property ndims:
        def __get__(self):
            """The number of dimensions in the layout"""
            return self.instance.num_dims()

cdef class LexicoLayout(Layout):

    def __cinit__(self, shape):
        self.instance = new core._LexicoLayout(shape)

    def __deallocate__(self):
        del self.instance

    def __init__(self, *args, **kwargs):
        pass

{% for typedef in typedefs %}
    {% with typedef=typedef %}

        {% include typedef.impl_template %}

    {% endwith %}
{% endfor %}