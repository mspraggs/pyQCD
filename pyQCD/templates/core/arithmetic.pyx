{% macro arithmetic_ops(operators, typename) %}
{% for funcname, op in zip(["add", "sub", "mul", "div"], "+-*/") %}
{% set ops = operators[(typename, funcname)] %}
    def __{{ funcname }}__(self, other):
        if isinstance(self, scalar_types):
            self = float(self)
        if isinstance(other, scalar_types):
            other = float(other)
        if isinstance(self, complex_types):
            self = Complex(self.real, self.imag)
        if isinstance(other, complex_types):
            other = Complex(other.real, other.imag)
{% for ret, lhs, rhs, lhs_bcast, rhs_bcast in ops %}
        if type(self) is {{ lhs }} and type(other) is {{ rhs }}:
{% if lhs == "float" or lhs == "Complex" %}
            return (<{{ typename }}>other)._{{ funcname }}_{{ rhs }}_{{ lhs }}(<{{ lhs }}>self)
{% else %}
            return (<{{ typename }}>self)._{{ funcname }}_{{ lhs }}_{{ rhs }}(<{{ rhs }}>other)
{% endif %}
{% endfor %}
        raise TypeError("Unsupported operand types for {{ typename }}.__{{ funcname }}__: "
                        "{} and {}".format(type(self), type(other)))

{% for ret, lhs, rhs, lhs_bcast, rhs_bcast in ops %}
{% if lhs != "float" and lhs != "Complex" %}
    cdef {{ ret }} _{{ funcname }}_{{ lhs }}_{{ rhs }}({{ lhs }} self, {{ rhs }} other):
{% set lhs_op = "self.instance" + (".broadcast()" if lhs_bcast else "[0]") %}
{% set rhs_op = "other" + (".instance" if rhs != "float" else "") + (("[0]" if not rhs_bcast else ".broadcast()") if rhs not in ["Complex", "float"] else "") %}
{% set ret_cpp = ret|to_underscores + "." + ret %}
        out = {{ ret }}()
        cdef {{ ret_cpp }}* cpp_out = new {{ ret_cpp }}()
        cpp_out[0] = {{ lhs_op }} {{ op }} {{ rhs_op }}
        out.instance = cpp_out
        return out

{% endif %}
{% endfor %}

{% endfor %}
{% endmacro %}