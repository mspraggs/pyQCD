{% if not typedef.is_static %}
    cdef int view_count
{% endif %}
{% if "Lattice" in typedef.structure %}
    cdef Layout layout
{% endif %}
{% if typedef.structure[0] == "Matrix" %}
    shape = {{ typedef.matrix_shape }}
{% else %}

    @property
    def shape(self):
        return {{ typedef.shape_expr }}

{% endif %}
{% if typedef.structure[0] == "Array" %}
    @property
    def size(self):
        return self.instance.size()

{% elif typedef.structure[0] == "Lattice" %}
    @property
    def volume(self):
        return self.instance.volume()

    @property
    def lattice_shape(self):
        return tuple(self.instance.lattice_shape())

    @property
    def num_dims(self):
        return self.instance.num_dims()

{% endif %}
{% if typedef.structure[0] == "Matrix" or typedef.structure[0] == "Array" %}
    def adjoint(self):
        cdef {{ typedef.name }} out = {{ typedef.name }}()
        out.instance[0] = self.instance.adjoint()
        return out

{% endif %}
{% for funcname in funcnames %}
    @staticmethod
    def {{ funcname }}({{ argstring }}):
        cdef {{ typedef.name }} out = {{ typedef.name }}({{ argpass }})
        {{ static_assign_line.format(funcname) }}
        return out

{% endfor %}