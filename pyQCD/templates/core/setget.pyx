{% if typedef.structure[0] == "Matrix" %}

    cdef validate_indices(self, unsigned int i{% if len(typedef.shape) == 2 %}, unsigned int j{% endif %}):
        if i >= {{ typedef.matrix_shape[0] }} or i < 0{% if len(typedef.matrix_shape) == 2 %} or j >= {{ typedef.matrix_shape[1] }} or j < 0{% endif %}:
            raise IndexError("Invalid index for type {{ typedef.name }}: {}".format((i{% if len(typedef.matrix_shape) == 2 %}, j{% endif %})))

    def __getitem__(self, index):
        out = Complex(0.0, 0.0)
        if type(index) is tuple:
{% if typedef.is_matrix %}
            self.validate_indices(index[0], index[1])
            out.instance = self.instance[0](index[0], index[1])
{% else %}
            self.validate_indices(index[0])
            out.instance = self.instance[0][index[0]]
        elif type(index) is int:
            self.validate_indices(index)
            out.instance = self.instance[0][index]
{% endif %}
        else:
            raise TypeError("Invalid index type in {{ typedef.name }}.__setitem__: "
                            "{}".format(type(index)))
        return out.to_complex()

    def __setitem__(self, index, value):
        if type(value) is Complex:
            pass
        elif hasattr(value, 'real') and hasattr(value, 'imag'):
            value = Complex(<{{ precision }}?>(value.real),
                            <{{ precision }}?>(value.imag))
        else:
            value = Complex(<{{ precision }}?>value, 0.0)
        if type(index) is tuple:
{% if typedef.is_matrix %}
            self.validate_indices(index[0], index[1])
            self.assign_elem(index[0], index[1], (<Complex>value).instance)
{% else %}
            self.validate_indices(index[0])
            self.assign_elem(index[0], (<Complex>value).instance)
        elif type(index) is int:
            self.validate_indices(index)
            self.assign_elem(index, (<Complex>value).instance)
{% endif %}
        else:
            raise TypeError("Invalid index type in {{ typedef.name }}.__setitem__: "
                            "{}".format(type(index)))

{% if typedef.is_matrix %}
    cdef void assign_elem(self, int i, int j, complex.Complex value):
        {{ typedef.cmodule }}.mat_assign(self.instance, i, j, value)
{% else %}
    cdef void assign_elem(self, int i, complex.Complex value):
        cdef complex.Complex* z = &(self.instance[0][i])
        z[0] = value
{% endif %}

{% elif typedef.structure[0] == "Array" %}

    def __getitem__(self, index):
        if type(index) is tuple and len(index) is 1:
            self.validate_index(index[0])
            out = {{ typedef.element_type.name }}()
            (<{{ typedef.element_type.name }}>out).instance[0] = (self.instance[0])[<int?>(index[0])]
            return out
        elif type(index) is tuple:
            self.validate_index(index[0])
            out = Complex(0.0, 0.0)
{% if len(typedef.matrix_shape) == 2 %}
            if index[1] > {{ typedef.matrix_shape[0] - 1 }} or index[1] < 0 or index[2] > {{ typedef.matrix_shape[1] - 1 }} or index[2] < 0:
                raise IndexError("Indices in {{ matrix_name }} element access out of bounds: "
                                 "{}".format(index))
            (<Complex>out).instance = self.instance[0][<int?>index[0]](<int?>index[1], <int?>index[2])
{% else %}
            if index[1] > {{ typedef.matrix_shape[0] - 1 }} or index[1] < 0:
                raise IndexError("Indices in {{ typedef.name }} element access out of bounds: "
                                 "{}".format(index))
            (<Complex>out).instance = self.instance[0][<int?>index[0]][<int?>index[1]]
{% endif %}
            return out.to_complex()
        else:
            self.validate_index(index)
            out = {{ typedef.element_type.name }}()
            (<{{ typedef.element_type.name }}>out).instance[0] = self.instance[0][<int?>index]
            return out

    def __setitem__(self, index, value):
        if type(value) is {{ typedef.element_type.name }}:
            self.validate_index(index[0] if type(index) is tuple else index)
            self.assign_elem(index[0] if type(index) is tuple else index, (<{{ typedef.element_type.name }}>value).instance[0])
            return
        elif type(value) is Complex:
            pass
        elif hasattr(value, "real") and hasattr(value, "imag") and isinstance(index, tuple):
            value = Complex(value.real, value.imag)
        else:
            value = Complex(<{{ precision }}?>value, 0.0)

        cdef {{ typedef.element_type.cmodule }}.{{ typedef.element_type.cname }}* mat = &(self.instance[0][<int?>index[0]])
{% if typedef.element_type.is_matrix %}
        if index[1] > {{ typedef.matrix_shape[0] - 1 }} or index[1] < 0 or index[2] > {{ typedef.matrix_shape[1] - 1 }} or index[2] < 0:
            raise IndexError("Indices in {{ typedef.name }} element access out of bounds: "
                             "{}".format(*index))
        {{ typedef.element_type.cmodule }}.mat_assign(mat, <int?>index[1], <int?>index[2], (<Complex?>value).instance)
{% else %}
        if index[1] > {{ typedef.matrix_shape[0] - 1}} or index[1] < 0:
            raise IndexError("Indices in {{ typedef.name }} element access out of bounds: "
                             "{}".format(*index))
        cdef complex.Complex* z = &(mat[0][<int?>index[1]])
        z[0] = (<Complex>value).instance
{% endif %}

    cdef void assign_elem(self, int i, {{ typedef.element_type.cmodule }}.{{ typedef.element_type.cname }} value):
        cdef {{ typedef.element_type.cmodule }}.{{ typedef.element_type.cname }}* m = &(self.instance[0][i])
        m[0] = value

{% elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Matrix" %}

    cdef validate_index(self, index):
        cdef unsigned int i
        if type(index) is tuple:
            for i in range(self.instance.num_dims()):
                if index[i] >= self.instance.lattice_shape()[i] or index[i] < 0:
                    raise IndexError("Index in {{ lattice_matrix_name }} element access "
                                     "out of bounds: {}".format(index))
        elif type(index) is int:
            if index < 0 or index >= self.instance.volume():
                raise IndexError("Index in {{ lattice_matrix_name }} element access "
                                 "out of bounds: {}".format(index))

    def __getitem__(self, index):
        cdef int num_dims = self.instance.num_dims()
        if type(index) is tuple and len(index) == self.instance.num_dims():
            out = {{ typedef.element_type.name }}()
            self.validate_index(index)
            (<{{ typedef.element_type.name }}>out).instance[0] = (<{{ typedef.name }}>self).instance[0](<vector[unsigned int]>index)
            return out
        if type(index) is tuple and len(index) == num_dims + {{ len(typedef.element_type.shape) }}:
            out = Complex(0.0, 0.0)
            self.validate_index(index)
            if index[num_dims] > {{ typedef.matrix_shape[0] - 1 }} or index[self.instance.num_dims()] < 0{% if typedef.matrix_shape[1] %} or index[self.instance.num_dims() + 1] > {{ typedef.matrix_shape[1] - 1 }} or index[self.instance.num_dims() + 1] < 0{% endif %}:
                raise IndexError("Indices in {{ typedef.name }} element access out of bounds: "
                                 "{}".format(index))
{% if typedef.element_type.is_matrix %}
            (<Complex>out).instance = (<{{ typedef.name }}>self).instance[0](<vector[unsigned int]>index[:num_dims])(index[num_dims], index[num_dims + 1])
{% else %}
            (<Complex>out).instance = (<{{ typedef.name }}>self).instance[0](<vector[unsigned int]>index[:num_dims])[index[num_dims]]
{% endif %}
            return out.to_complex()
        if type(index) is int:
            out = {{ typedef.element_type.name }}()
            self.validate_index(index)
            (<{{ typedef.element_type.name }}>out).instance[0] = (<{{ typedef.name }}>self).instance[0](<int>index)
            return out
        raise TypeError("Invalid index type in {{ typedef.name }}.__getitem__")

    def __setitem__(self, index, value):
        cdef int num_dims = self.instance.num_dims()
        if type(value) is {{ typedef.element_type.name }}:
            self.validate_index(index[:num_dims] if type(index) is tuple else index)
            self.assign_elem(index[:num_dims] if type(index) is tuple else index, (<{{ typedef.element_type.name }}>value).instance[0])
            return
        elif type(value) is Complex:
            pass
        elif hasattr(value, "real") and hasattr(value, "imag") and isinstance(index, tuple):
            value = Complex(value.real, value.imag)
        else:
            value = Complex(<{{ precision }}?>value, 0.0)

        cdef {{ typedef.element_type.cmodule }}.{{ typedef.element_type.cname }}* mat
        if type(index) is tuple:
            mat = &(self.instance[0](<vector[unsigned int]?>index[:num_dims]))
        else:
            mat = &(self.instance[0](<int?>index))
{% if typedef.element_type.is_matrix %}
        if index[num_dims] > {{ typedef.matrix_shape[0] - 1}} or index[num_dims] < 0 or index[num_dims + 1] > {{ typedef.matrix_shape[1] - 1 }} or index[num_dims + 1] < 0:
            raise IndexError("Indices in {{ matrix_name }} element access out of bounds: "
                             "{}".format(index))
        {{ typedef.element_type.cmodule }}.mat_assign(mat, <int?>index[num_dims], <int?>index[num_dims + 1], (<Complex?>value).instance)
{% else %}
        if index[num_dims] > {{ typedef.matrix_shape[0] - 1}} or index[num_dims] < 0:
            raise IndexError("Indices in {{ matrix_name }} element access out of bounds: "
                             "{}".format(index))
        cdef complex.Complex* z = &(mat[0][<int?>index[num_dims]])
        z[0] = (<Complex>value).instance
{% endif %}

    cdef assign_elem(self, index, {{ typedef.element_type.cmodule }}.{{ typedef.element_type.cname }} value):
        cdef {{ typedef.element_type.cmodule }}.{{ typedef.element_type.cname }}* m
        if type(index) is tuple:
            m = &(self.instance[0](<vector[unsigned int]>index))
        else:
            m = &(self.instance[0](<int?>index))
        m[0] = value

{% elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Array" %}

{% endif %}