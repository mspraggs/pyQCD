{# TODO: Refactor this so it's less of an eyesore #}
{% macro arithmetic_ops(operators, typename, scalar_types, operator_map, is_lattice) %}
{% for op, funcnames in operator_map.items() %}
{# Set a couple of flags to determine if we have a complex lhs or rhs in our
 # list of operators. #}
{% set lhs_complex = False %}
{% set rhs_complex = False %}
{% for funcname in funcnames %}
    def __{{ funcname }}__(self, other):
{# Loop through list of operations and use each item to write out type checking
 # code and function calls #}
{% for ret, lhs, rhs, lhs_bcast, rhs_bcast in operators[(typename, op)] %}
        if type(self) is {{ lhs }} and type(other) is {{ rhs }}:
{# If self is a scalar type (e.g. float, Complex, etc.), switch lhs and rhs to
 # call function where other is scalar #}
{% if lhs in scalar_types or lhs == "Complex" %}
            return (<{{ typename }}>other)._{{ funcnames[0] }}_{{ rhs }}_{{ lhs }}(<{{ lhs }}>self)
{% else %}
            return (<{{ typename }}>self)._{{ funcnames[0] }}_{{ lhs }}_{{ rhs }}(<{{ rhs }}>other)
{% endif %}
{% set lhs_complex = True if op in "*/" and lhs == "Complex" else lhs_complex %}
{% set rhs_complex = True if op in "*/" and rhs == "Complex" else rhs_complex %}
{% if loop.last %}
{# If we have complex types in the list of operations, tack on some code to
 # handle general complex data types, such as the Python complex type #}
{% if lhs_complex %}
        if hasattr(self, "real") and hasattr(self, "imag") and type(other) is {{ typename }}:
            return (<{{ typename }}>other)._{{ funcnames[0] }}_{{ typename }}_Complex(Complex(self.real, self.imag))
{% endif %}
{% if rhs_complex %}
        if type(self) is {{ typename }} and hasattr(other, "real") and hasattr(other, "imag"):
            return (<{{ typename }}>self)._{{ funcnames[0] }}_{{ typename }}_Complex(Complex(other.real, other.imag))
{% endif %}
        raise TypeError("Unsupported operand types for {{ typename }}.__{{ funcname }}__: "
                        "{} and {}".format(type(self), type(other)))
{% endif %}
{% endfor %}

{% endfor %}
{# Now write out the C functions to handle the above operations #}
{% for ret, lhs, rhs, lhs_bcast, rhs_bcast in operators[(typename, op)] %}
{# Don't bother with functions where self is scalar, as this is basically code duplication #}
{% if lhs not in scalar_types and lhs != "Complex" %}
    cdef inline {{ ret }} _{{ funcnames[0] }}_{{ lhs }}_{{ rhs }}({{ lhs }} self, {{ rhs }} other):
{% set lhs_op = "self.instance" + (".broadcast()" if lhs_bcast else "[0]") %}
{% set rhs_op = "other" + (".instance" if rhs not in scalar_types else "") + (("[0]" if not rhs_bcast else ".broadcast()") if rhs != "Complex" and rhs not in scalar_types else "") %}
        cdef {{ ret }} out = {{ ret }}({% if is_lattice %}self.layout{% endif %})
        out.instance[0] = {{ lhs_op }} {{ op }} {{ rhs_op }}
        return out

{% endif %}
{% endfor %}
{% endfor %}
{% endmacro %}