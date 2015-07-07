{% if typedef.structure[0] == "Matrix" %}
    def __cinit__(self):
        self.instance = new {{ typedef.cmodule }}.{{ typedef.cname }}()

    def __init__(self, *args):
        cdef int i, j
        if not args:
            pass
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            for i, elem in enumerate(args[0]):
{% if typedef.is_matrix %}
                for j, subelem in enumerate(elem):
                    self.validate_indices(i, j)
                    self[i, j] = subelem
{% else %}
                self.validate_indices(i)
                self[i] = elem
{% endif %}

    def __dealloc__(self):
        del self.instance

{% elif typedef.structure[0] == "Array" %}

    cdef _init_with_args_(self, unsigned int N, {{ typedef.element_type.name }} value):
        self.instance[0] = {{ typedef.cmodule }}.{{ typedef.cname }}(N, value.instance[0])

    cdef validate_index(self, int i):
        if i >= self.instance.size() or i < 0:
            raise IndexError("Index in {{ array_name }} element access out of bounds: "
                             "{}".format(i))

    def __cinit__(self):
        self.instance = new {{ typedef.cmodule }}.{{ typedef.cname }}()
        self.view_count = 0

    def __init__(self, *args):
        cdef int i, N
        if not args:
            pass
        elif len(args) == 1 and hasattr(args[0], "__len__"):
            N = len(args[0])
            self.instance.resize(N)
            for i in range(N):
                self[i] = {{ typedef.element_type.name }}(args[0][i])
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], {{ typedef.element_type.name }}):
            self._init_with_args_(args[0], args[1])
        else:
            raise TypeError("{{ array_name }} constructor expects "
                            "either zero or two arguments")

    def __dealloc__(self):
        del self.instance

{% elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Matrix" %}

    def __cinit__(self, Layout layout, *args):
        self.instance = new {{ typedef.cmodule }}.{{ typedef.cname }}(layout.instance[0], {{ typedef.element_type.cmodule }}.{{ typedef.element_type.cname }}())
        self.layout = layout
        self.view_count = 0

    def __init__(self, Layout layout, *args):
        cdef int i, volume
        volume = layout.instance.volume()
        if len(args) is 1 and type(args[0]) is {{ typedef.element_type.name }}:
            for i in range(volume):
                self.instance[0][i] = (<{{ typedef.element_type.name }}>args[0]).instance[0]

    def __dealloc__(self):
        del self.instance

{% elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Array" %}

{% endif %}
