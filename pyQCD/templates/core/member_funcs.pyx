{% set is_square = typedef.matrix_shape[0] == typedef.matrix_shape[1] %}
{% if typedef.structure[0] == "Matrix" %}
    shape = {{ typedef.matrix_shape }}

    def adjoint(self):
        cdef {{ typedef.name }} out = {{ typedef.name }}()
        out.instance[0] = self.instance.adjoint()
        return out

{% for funcname in ["zeros", "ones"] + (["identity"] if is_square else []) %}
    @staticmethod
    def {{ funcname }}():
        out = {{ typedef.name }}()
        out.instance[0] = {{ typedef.cmodule }}.{{ funcname }}()
        return out

{% endfor %}
{% elif typedef.structure[0] == "Array" %}
    cdef int view_count

    @property
    def size(self):
        return self.instance.size()

    @property
    def shape(self):
        return (self.size,) + {{ typedef.matrix_shape }}

    def adjoint(self):
        cdef {{ typedef.name }} out = {{ typedef.name }}()
        out.instance[0] = self.instance.adjoint()
        return out

{% for funcname in ["zeros", "ones"] + (["identity"] if is_square else []) %}
    @staticmethod
    def {{ funcname }}(int size):
        out = {{ typedef.name }}()
        out.instance[0] = {{ typedef.cmodule }}.{{ typedef.cname }}(size, {{ typedef.element_type.cmodule }}.{{ funcname }}())
        return out

{% endfor %}
{% elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Matrix" %}
    cdef Layout layout
    cdef int view_count

    @property
    def volume(self):
        return self.instance.volume()

    @property
    def lattice_shape(self):
        return tuple(self.instance.lattice_shape())

    @property
    def shape(self):
        return tuple(self.instance.lattice_shape()) + {{ typedef.matrix_shape }}

    @property
    def num_dims(self):
        return self.instance.num_dims()

{% for funcname in ["zeros", "ones"] + (["identity"] if is_square else []) %}
    @staticmethod
    def {{ funcname }}(Layout layout):
        elem = {{ typedef.element_type.cmodule }}.{{ funcname }}()
        out = {{ typedef.name }}(layout)
        out.instance[0] = {{ typedef.cmodule }}.{{ typedef.cname }}(layout.instance[0], {{ typedef.element_type.cmodule }}.{{ funcname }}())
        return out

{% endfor %}
{% elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Array" %}

{% endif %}