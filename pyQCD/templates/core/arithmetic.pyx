{% for op in operator_map %}
{% set funcnames = operator_map[op] %}
{% for funcname in funcnames %}
    def __{{ funcname }}__(self, other):
{% for ret, lhs, rhs, bcast in operations[op] %}
        if type(self) is {{ lhs.name }} and type(other) is {{ rhs.name }}:
{% if lhs.builtin or lhs.name == "Complex" %}
            return (<{{ rhs.name }}>other)._{{ funcnames[0] }}_{{ rhs.name }}_{{ lhs.name }}(<{{ lhs.name }}>self)
{% else %}
            return (<{{ lhs.name }}>self)._{{ funcnames[0] }}_{{ lhs.name }}_{{ rhs.name }}(<{{ rhs.name }}>other)
{% endif %}
{% endfor %}
{% if lhs_complex[op] %}
        if hasattr(self, "real") and hasattr(self, "imag") and type(other) is {{ typedef.name }}:
            return (<{{ typedef.name }}>other)._{{ funcnames[0] }}_{{ typedef.name }}_Complex(Complex(self.real, self.imag))
{% endif %}
{% if rhs_complex[op] %}
        if type(self) is {{ typedef.name }} and hasattr(other, "real") and hasattr(other, "imag"):
            return (<{{ typedef.name }}>self)._{{ funcnames[0] }}_{{ typedef.name }}_Complex(Complex(other.real, other.imag))
{% endif %}
        raise TypeError("Unsupported operand types for {{ typedef.name }}.__{{ funcname }}__: "
                        "{} and {}".format(type(self), type(other)))

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