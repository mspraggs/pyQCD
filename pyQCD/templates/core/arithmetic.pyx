{% for op in operator_map %}
{% set funcnames = operator_map[op] %}
{% for funcname in funcnames %}
    def __{{ funcname }}__(self, other):
{% for ret, lhs, rhs, bcast in operations[op] %}
        if type(self) is {{ lhs.name }} and type(other) is {{ rhs.name }}:
{% if lhs.builtin or lhs.name == "Complex" %}
            return (<{{ rhs.name }}>self)._{{ funcnames[0] }}_{{ rhs.name }}_{{ lhs.name }}(<{{ lhs.name }}>other)
{% else %}
            return (<{{ lhs.name }}>self)._{{ funcnames[0] }}_{{ lhs.name }}_{{ rhs.name }}(<{{ rhs.name }}>other)
{% endif %}
{% endfor %}
{% endfor %}

{% for ret, lhs, rhs, bcast in operations[op] %}
{% if not lhs.builtin and lhs.name != "Complex" %}
    cdef inline {{ ret.name }} _{{ funcnames[0] }}_{{ lhs.name }}_{{ rhs.name }}({{ lhs.name }} self, {{ rhs.name }} other):
        cdef {{ ret.name }} out = {{ ret.name }}({% if "Lattice" in ret.structure %}self.layout{% endif %})
        out.instance[0] = {{ lhs.accessor("self", bcast == 'L') }} {{ op }} {{ rhs.accessor("other", bcast == 'R') }}
        return out

{% endif %}
{% endfor %}
{% endfor %}