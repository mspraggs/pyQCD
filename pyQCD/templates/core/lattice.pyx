{% set elem_type = typedef.element_type %}
{% set is_matrix = typedef.ndims == 2 %}
cdef class {{ typedef.name }}:
    """Lattice colour {% if is_matrix %}matrix{% else %}vector{% endif %} of specified shape.

    A {{ typedef.name }} instance is initialised with the specified lattice
    shape, with the specified number of colour {% if is_matrix %}matrices{% else %}vectors{% endif %} at each site.

    Supports indexing and attribute lookup akin to the numpy.ndarray type.

    Args:
      shape (tuple-like): The shape of the lattice.
      site_size (int): The number of colour {% if is_matrix %}matrices{% else %}vectors{% endif %} at each site.
    """

    def __cinit__(self, shape, int site_size=1):
        """Constructor for {{ typedef.name }} type. See help({{ typedef.name }})."""
        self.lexico_layout = new LexicoLayout(shape)
        self.view_count = 0
        self.site_size = site_size
        self.instance = new _{{ typedef.cname }}(self.lexico_layout[0],  _{{ typedef.element_type.cname }}(_{{ elem_type.cname }}_zeros()), site_size)

    def __dealloc__(self):
        del self.instance
        del self.lexico_layout

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(atomics.Complex)

        {% set last_dim = elem_type.shape[1] if is_matrix else elem_type.shape[0] %}
        self.buffer_shape[0] = self.instance[0].volume() * self.site_size
        self.buffer_strides[0] = itemsize * {{ elem_type.size }}
        self.buffer_shape[1] = {{ last_dim }}
        self.buffer_strides[1] = itemsize
        {% if elem_type.ndims == 2 %}
        self.buffer_shape[2] = {{ elem_type.shape[1] }}
        self.buffer_strides[2] = itemsize * {{ elem_type.shape[0] }}
        {% endif %}

        buffer.buf = <char*>&(self.instance[0][0])

        {% set num_format = "dd" if precision == "double" else "ff" %}
        buffer.format = "{{ num_format }}"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = itemsize * {{ elem_type.size }} * self.instance[0].volume() * self.site_size
        buffer.ndim = {{ elem_type.ndims + 1 }}

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
            return out.reshape(tuple(self.lexico_layout.shape()) + (self.site_size,) + {{ elem_type.shape }})

        def __set__(self, value):
            out = np.asarray(self)
            out.dtype = complex
            out = out.reshape(tuple(self.lexico_layout.shape()) + (self.site_size,) + {{ elem_type.shape }})
            out[:] = value

    def __repr__(self):
        return self.as_numpy.__repr__()
