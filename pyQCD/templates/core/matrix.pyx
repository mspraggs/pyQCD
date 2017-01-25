cdef class {{ typedef.name }}:

    def __cinit__(self):
        self.instance = new _{{ typedef.cname }}(core._{{ typedef.cname }}_zeros())
        self.view_count = 0

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(atomics.Complex)

	{% set last_dim = typedef.shape[1] if typedef.ndims == 2 else typedef.shape[0] %}
        self.buffer_shape[0] = {{ last_dim }}
        self.buffer_strides[0] = itemsize
        {% if typedef.ndims == 2 %}
        self.buffer_shape[1] = {{ typedef.shape[0] }}
        self.buffer_strides[1] = itemsize * {{ typedef.shape[0] }}
        {% endif %}

        buffer.buf = <char*>self.instance

        {% set num_format = "dd" if precision == "double" else "ff" %}
        buffer.format = "{{ num_format }}"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * {{ typedef.size }}
        buffer.ndim = {{ typedef.ndims }}

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        self.view_count -= 1

    property as_numpy:
        """Return a view to this object as a numpy array"""
        def __get__(self):
            out = np.asarray(self)
            out.dtype = complex
            return out.reshape({{ typedef.shape }})

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out = out.reshape({{ typedef.shape }})
            out[:] = value