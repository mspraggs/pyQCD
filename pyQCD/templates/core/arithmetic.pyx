{% for op in operator_map %}
{% set funcnames = operator_map[op] %}
{% for funcname in funcnames %}
    def __{{ funcname }}__(self, other):
{% for ret, lhs, rhs, bcast in operations[op] %}
        if type(self) is {{ lhs.name }} and type(other) is {{ rhs.name }}:
            return (<{{ lhs.name }}>self)._{{ funcnames[0] }}_{{ lhs.name }}_{{ rhs.name }}(<{{ rhs.name }}>other)
{% endfor %}

{% for ret, lhs, rhs, bcast in operations[op] %}
    cdef inline {{ ret.name }} _{{ funcnames[0] }}_{{ lhs.name }}_{{ rhs.name }}({{ lhs.name }} self, {{ rhs.name }} other):
        cdef {{ ret.name }} out = {{ ret.name }}({% if "Lattice" in ret.structure %}self.layout{% endif %})
        out.instance[0] = {{ lhs.accessor("self", bcast == 'L') }} {{ op }} {{ rhs.accessor("other", bcast == 'R') }}
        return out

{% endfor %}
{% endfor %}
{% endfor %}