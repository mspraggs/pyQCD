    cdef int view_count
    cdef Py_ssize_t buffer_shape[{{ typedef.buffer_ndims }}]
    cdef Py_ssize_t buffer_strides[{{ typedef.buffer_ndims }}]

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

{% for shape, stride in buffer_info %}
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