    cdef {{typedef.cmodule }}.{{ typedef.cname }}* instance
    cdef Layout layout

    def __cinit__(self, {{ argstring }}):
        self.instance = new {{ constructor_call }}
{% if "layout" in argstring %}
        self.layout = layout
{% endif %}

    def __dealloc__(self):
        del self.instance