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
        raise TypeError("Unsupported operand types for {{ typedef.name }}.__{{ funcname }}__: "
                        "{} and {}".format(type(self), type(other)))

{% endfor %}
{% for ret, lhs, rhs, bcast in operations[op] %}
    cdef inline {{ ret.name }} _{{ funcnames[0] }}_{{ lhs.name }}_{{ rhs.name }}({{ lhs.name }} self, {{ rhs.name }} other):
{% set layout_operand = "self" if "Lattice" in rhs.structure else ("other" if "Lattice" in lhs.structure else "") %}
        cdef {{ ret.name }} out = {{ ret.name }}({% if "Lattice" in ret.structure %}{{ layout_operand }}.layout, {% endif %}{% if "Array" in ret.structure %}1{% endif %})
        out.instance[0] = {{ lhs.accessor("self", False) }} {{ op }} {{ rhs.accessor("other", False) }}
        return out

{% endfor %}
{% endfor %}