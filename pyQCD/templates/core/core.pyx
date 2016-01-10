from cpython cimport Py_buffer
from libcpp.vector cimport vector

import numpy as np

cimport complex
cimport layout
from operators cimport *
{% for typedef in typedefs %}
cimport {{ typedef.cmodule }}
{% endfor %}

scalar_types = (int, float, np.single, np.double,
                np.float16, np.float32, np.float64, np.float128)
complex_types = (complex, np.complex, np.complex64, np.complex128,
                 np.complex256)

ctypedef {{ precision }} Real

cdef class Complex:
    cdef complex.Complex instance

    def __init__(self, x, y):
        self.instance = complex.Complex(x, y)

    @property
    def real(self):
        return self.instance.real()

    @property
    def imag(self):
        return self.instance.imag()

    def to_complex(self):
        return complex(self.instance.real(), self.instance.imag())

    def __repr__(self):
        if self.real == 0:
            return "{}j".format(self.imag)
        else:
            return "({}{}{}j)".format(self.real,
                                      ('+' if self.imag >= 0 else ''),
                                      self.imag)


cdef class Layout:
    cdef layout.Layout* instance

    def get_array_index(self, site_label):

        if type(site_label) is int:
            return self.instance.get_array_index(<unsigned int>site_label)
        if type(site_label) is list or type(site_label) is tuple:
            return self.instance.get_array_index(<vector[unsigned int]>site_label)
        raise TypeError("Unknown type in Layout.get_array_index: {}"
                        .format(type(site_label)))

    def get_site_index(self, unsigned int array_index):
        return self.instance.get_site_index(array_index)

    @property
    def num_dims(self):
        return self.instance.num_dims()

    @property
    def volume(self):
        return self.instance.volume()

    @property
    def shape(self):
        return tuple(self.instance.shape())


cdef class LexicoLayout(Layout):

    def __init__(self, shape):
        self.instance = <layout.Layout*>new layout.LexicoLayout(<vector[unsigned int]?>shape)

    def __dealloc__(self):
        del self.instance


{% for typedef in typedefs %}
cdef class {{ typedef.name }}:
    cdef {{typedef.cmodule }}.{{ typedef.cname }}{% if typedef.wrap_ptr %}*{% endif %} instance
{% for type, name, init in typedef.cmembers %}
    cdef {{ type }} {{ name }}
{% endfor %}

    def __cinit__(self, {{ typedef.ctor_argstring }}):
        self.instance = new {{ typedef.init_code }}
{% for type, name, init in typedef.cmembers %}
{% if init %}
        self.{{ name }} = {{ init }}
{% endif %}
{% endfor %}

    def __dealloc__(self):
        del self.instance
        {% for type, name, init in typedef.cmembers %}
        {% if "*" in type %}
        del self.{{ name }}
        {% endif %}
        {% endfor %}

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

{% set buffer_iter, buffer_size = typedef.buffer_info("itemsize") %}
{% for shape, stride in buffer_iter %}
        self.buffer_shape[{{ loop.index0 }}] = {{ stride }}
        self.buffer_strides[{{ loop.index0 }}] = {{ shape }}
{% endfor %}

        buffer.buf = {% if typedef.is_static %}<char*>self.instance{% else %}<char*>&(self.instance[0][0]){% endif %}

{% set num_format = "d" if precision == "double" else "f" %}
        buffer.format = "{{ num_format * 2 }}"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = {{ buffer_size }}
        buffer.ndim = {{ typedef.buffer_ndims }}

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL
        {% if not typedef.is_static %}

        self.view_count += 1
        {% endif %}

    def __releasebuffer__(self, Py_buffer* buffer):
{% if not typedef.is_static %}
        self.view_count -= 1{% else %}
        pass
{% endif %}

    property as_numpy:
        """Return a view to this object as a numpy array"""
        def __get__(self):
            out = np.asarray(self)
            out.dtype = complex
            return out

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out[:] = value

{% set operations = typedef.generate_arithmetic_operations(typedefs) %}
{% for op in operator_map %}
{% for funcname in operator_map[op] %}
    def __{{ funcname }}__(self, other):
{% for ret, lhs, rhs in operations[op] %}
        if type(self) is {{ lhs.name }} and type(other) is {{ rhs.name }}:
            return (<{{ lhs.name }}>self)._{{ operator_map[op][0] }}_{{ lhs.name }}_{{ rhs.name }}(<{{ rhs.name }}>other)
{% endfor %}
        raise TypeError("Unsupported operand types for {{ typedef.name }}.__{{ funcname }}__: "
                        "{} and {}".format(type(self), type(other)))

{% endfor %}
{% for ret, lhs, rhs in operations[op] %}
    cdef inline {{ ret.name }} _{{ operator_map[op][0] }}_{{ lhs.name }}_{{ rhs.name }}({{ lhs.name }} self, {{ rhs.name }} other):
{% set layout_operand = "self" if "Lattice" in rhs.structure else ("other" if "Lattice" in lhs.structure else "") %}
        cdef {{ ret.name }} out = {{ ret.name }}({% if "Lattice" in ret.structure %}{{ layout_operand }}.layout{% endif %})
        out.instance[0] = {{ lhs.accessor("self", False) }} {{ op }} {{ rhs.accessor("other", False) }}
        return out

{% endfor %}
{% endfor %}
{% endfor %}