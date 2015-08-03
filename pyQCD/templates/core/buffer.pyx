    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

{% set buffer_ndim = typedef.buffer_ndim %}
{% set size_exprs = typedef.size_exprs %}
{% set i = 0 %}
{% if typedef.structure[0] == "Matrix" %}
        self.buffer_shape[0] = {{ num_rows }}
        self.buffer_strides[0] = itemsize
{% if is_matrix %}
        self.buffer_shape[1] = {{ num_cols }}
        self.buffer_strides[1] = {% if num_cols > 1 %}{{ num_cols }} * {% endif %}itemsize
{% endif %}
{% elif typedef.structure[0] == "Array" %}
        self.buffer_shape[0] = self.instance.size()
        self.buffer_strides[0] = itemsize * {{ num_cols * num_rows }}
        self.buffer_shape[1] = {{ num_rows }}
        self.buffer_strides[1] = itemsize
{% if is_matrix %}
        self.buffer_shape[2] = {{ num_cols }}
        self.buffer_strides[2] = {% if num_cols > 1 %}{{ num_cols }} * {% endif %}itemsize
{% endif %}
{% elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Matrix" %}
        self.buffer_shape[0] = self.instance.volume()
        self.buffer_strides[0] = itemsize * {{ num_cols * num_rows }}
        self.buffer_shape[1] = {{ num_rows }}
        self.buffer_strides[1] = itemsize
{% if is_matrix %}
        self.buffer_shape[2] = {{ num_cols }}
        self.buffer_strides[2] = {% if num_cols > 1 %}{{ num_cols }} * {% endif %}itemsize
{% endif %}
{% elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Array" %}
        self.buffer_shape[0] = self.instance.volume()
        self.buffer_strides[0] = itemsize * {{ num_cols * num_rows }} * self.instance[0][0].size()
        self.buffer_shape[1] = self.instance[0][0].size()
        self.buffer_strides[1] = itemsize * {{ num_cols * num_rows }}
        self.buffer_shape[2] = {{ num_rows }}
        self.buffer_strides[2] = itemsize
{% if is_matrix %}
        self.buffer_shape[3] = {{ num_cols }}
        self.buffer_strides[3] = {% if num_cols > 1 %}{{ num_cols }} * {% endif %}itemsize
{% endif %}
{% endif %}

        buffer.buf = {% if typedef.is_static %}<char*>self.instance{% else %}<char*>&(self.instance[0][0]){% endif %}

{% set num_format = "d" if precision == "double" else "f" %}
        buffer.format = "{{ num_format * 2 }}"
        buffer.internal = NULL
        buffer.itemsize = itemsize
{% if typedef.structure[0] == "Matrix" %}
        buffer.len = {{ num_rows * num_cols }} * itemsize
{% elif typedef.structure[0] == "Array" %}
        buffer.len = {{ num_rows * num_cols }} * self.instance.size() * itemsize
{% elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Matrix" %}
        buffer.len = {{ num_rows * num_cols }} * self.instance.volume() * itemsize
{% elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Matrix" %}
        buffer.len = {{ num_rows * num_cols }} * self.instance.volume() * self.instance[0][0].size() * itemsize
{% endif %}
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