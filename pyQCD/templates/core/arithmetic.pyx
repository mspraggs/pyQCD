{% macro arithmetic_ops(operators, typename, scalar_types, operator_map) %}
{% for op, funcnames in operator_map.items() %}
{% set ops = operators[(typename, op)] %}
{% set lhs_complex = False %}
{% set rhs_complex = False %}
{% for funcname in funcnames %}
    def __{{ funcname }}__(self, other):
{% for ret, lhs, rhs, lhs_bcast, rhs_bcast in ops %}
        if type(self) is {{ lhs }} and type(other) is {{ rhs }}:
{% if lhs in scalar_types or lhs == "Complex" %}
            return (<{{ typename }}>other)._{{ funcnames[0] }}_{{ rhs }}_{{ lhs }}(<{{ lhs }}>self)
{% else %}
            return (<{{ typename }}>self)._{{ funcnames[0] }}_{{ lhs }}_{{ rhs }}(<{{ rhs }}>other)
{% endif %}
{% if op in "*/" and lhs == "Complex" %}
{% set lhs_complex = True %}
{% endif %}
{% if op in "*/" and rhs == "Complex" %}
{% set rhs_complex = True %}
{% endif %}
{% if loop.last %}
{% if lhs_complex %}
        if hasattr(self, "real") and hasattr(self, "imag") and type(other) is {{ typename }}:
            return (<{{ typename }}>other)._{{ funcnames[0] }}_{{ typename }}_Complex(Complex(self.real, self.imag))
{% endif %}
{% if rhs_complex %}
        if type(self) is {{ lhs }} and hasattr(other, "real") and hasattr(other, "imag"):
            return (<{{ typename }}>self)._{{ funcnames[0] }}_{{ typename }}_Complex(Complex(other.real, other.imag))
{% endif %}
        raise TypeError("Unsupported operand types for {{ typename }}.__{{ funcname }}__: "
                        "{} and {}".format(type(self), type(other)))
{% endif %}
{% endfor %}

{% endfor %}
{% for ret, lhs, rhs, lhs_bcast, rhs_bcast in ops %}
{% if lhs not in scalar_types and lhs != "Complex" %}
    cdef inline {{ ret }} _{{ funcnames[0] }}_{{ lhs }}_{{ rhs }}({{ lhs }} self, {{ rhs }} other):
{% set lhs_op = "self.instance" + (".broadcast()" if lhs_bcast else "[0]") %}
{% set rhs_op = "other" + (".instance" if rhs not in scalar_types else "") + (("[0]" if not rhs_bcast else ".broadcast()") if rhs != "Complex" and rhs not in scalar_types else "") %}
        cdef {{ ret }} out = {{ ret }}()
        out.instance[0] = {{ lhs_op }} {{ op }} {{ rhs_op }}
        return out

{% endif %}
{% endfor %}
{% endfor %}
{% endmacro %}