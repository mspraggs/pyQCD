{% set is_matrix = typedef.ndims == 2 %}
cdef class {{ typedef.name }}:
    """Statically-sized colour {% if is_matrix %}matrix{% else %}vector{% endif %} of shape {{ typedef.shape }}.

    Supports indexing and attribute lookup akin to the numpy.ndarray type.

    Attributes:
      as_numpy (numpy.ndarray): A numpy array view onto the underlying buffer
        containing the lattice data.
    """

    def __cinit__(self):
        """Constructor for {{ typedef.name }} type. See help({{ typedef.name }})."""
        self.instance = new _{{ typedef.cname }}(core._{{ typedef.cname }}_zeros())
        self.view_count = 0

    def __dealloc__(self):
        del self.instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(atomics.Complex)

	    {% set last_dim = typedef.shape[1] if is_matrix else typedef.shape[0] %}
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

    def __getitem__(self, index):
        return self.as_numpy[index]

    def __setitem__(self, index, value):
        if hasattr(value, 'as_numpy'):
            self.as_numpy[index] = value.as_numpy
        else:
            self.as_numpy[index] = value

    def __getattr__(self, attr):
        return getattr(self.as_numpy, attr)

    property as_numpy:
        def __get__(self):
            """numpy.ndarray: A numpy array view onto the underlying data buffer
            """
            out = np.asarray(self)
            out.dtype = complex
            return out.reshape({{ typedef.shape }})

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out = out.reshape({{ typedef.shape }})
            out[:] = value
{% if is_matrix %}

    @staticmethod
    def random():
        """Generate a random SU(N) {{ typedef.name }} instance with shape {{ typedef.shape }}."""
        ret = {{ typedef.name }}()
        ret.instance[0] = _random_colour_matrix()
        return ret
{% endif %}

    def __repr__(self):
        return self.as_numpy.__repr__()
