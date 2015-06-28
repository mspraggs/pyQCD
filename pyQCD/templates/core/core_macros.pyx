{% macro make_buffer_code(num_rows, num_cols, precision, is_matrix, is_array, is_lattice) %}
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(complex.Complex)

{% set ndim = 2 if is_matrix else 1 %}
{% set ndim = ndim + (1 if is_array else 0) + (1 if is_lattice else 0) %}
{% set shape = [] %}
{% set strides = [] %}
{% if is_array and is_lattice %}
{% set len_code = "self.instance.volume() * self.instance[0][0].size()" %}
{% do shape.append("self.instance.volume()") %}
{% do shape.append("self.instance[0][0].size()") %}
{% elif is_array %}
{% set len_code = "self.instance.size()" %}
{% do shape.append("self.instance.size()") %}
{% elif is_lattice %}
{% set len_code = "self.instance.volume()" %}
{% do shape.append("self.instance.volume()") %}
{% else %}
{% set len_code = "1" %}
{% endif %}
{% for s in shape %}
        self.buffer_shape[{{ loop.index0 }}] = {{ s }}
        self.buffer_strides[{{ loop.index0 }}] = {{ shape[loop.index0 + 1:]|join(" * ") }}{% if loop.index0 > 0 %} * {% endif %}{{ num_cols * num_rows }} * itemsize
{% endfor %}
        self.buffer_shape[{{ len(shape) }}] = {{ num_rows }}
        self.buffer_strides[{{ len(shape) }}] = itemsize
        {% if is_matrix %}
        self.buffer_shape[{{ len(shape) + 1 }}] = {{ num_cols }}
        self.buffer_strides[{{ len(shape) + 1 }}] = {{ num_cols }} * itemsize
        {% endif %}

        buffer.buf = {% if not (is_array or is_lattice) %}<char*>self.instance{% else %}<char*>&(self.instance[0][0]){% endif %}

        {% set num_format = "d" if precision == "double" else "f" %}
        buffer.format = "{{ num_format + num_format }}"
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = {{ num_rows * num_cols }} * {{ len_code }} * itemsize
        buffer.ndim = {{ ndim }}

        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.buffer_shape
        buffer.strides = self.buffer_strides
        buffer.suboffsets = NULL
        {% if is_array or is_lattice %}

        self.view_count += 1
        {% endif %}

    def __releasebuffer__(self, Py_buffer* buffer):
        {% if is_array or is_lattice %}
        self.view_count -= 1{% else %}
        pass{% endif %}
{% endmacro %}
