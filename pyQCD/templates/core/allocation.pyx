    cdef {{typedef.cmodule }}.{{ typedef.cname }}* instance

{% for funcname in ["cinit", "init"] %}
    def __{{ funcname }}__(self, {{ argstring }}):
        self.instance = new {{ constructor_call }}

{% endfor %}
    def __dealloc__(self):
        del self.instance