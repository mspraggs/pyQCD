from cpython cimport Py_buffer

cimport layout
from operators cimport *
{% for typedef in typedefs %}
cimport {{ typedef.cmodule }}
{% endfor %}


{% for typedef in typedefs %}
cdef class {{ typedef.name }}:
    cdef {{typedef.cmodule }}.{{ typedef.cname }}{% if typedef.wrap_ptr %}*{% endif %} instance
    {% for type, name, init in typedef.cmembers %}
    cdef {{ type }} {{ name }}
    {% endfor %}

    {% set operations = typedef.generate_arithmetic_operations(typedefs) %}
    {% for op in operator_map %}
        {% for ret, lhs, rhs in operations[op] %}
    cdef inline {{ ret.name }} _{{ operator_map[op][0] }}_{{ lhs.name }}_{{ rhs.name }}({{ lhs.name }} self, {{ rhs.name }} other):
            {% set lattice_name = rhs.get_lattice_name(lhs, "self", "other") %}
            {% set constructor_args = "{}.lexico_layout.shape(), self.site_size".format(lattice_name) if lattice_name else "" %}
        cdef {{ ret.name }} out = {{ ret.name }}({{ constructor_args }})
        out.instance[0] = {{ lhs.accessor("self", False) }} {{ op }} {{ rhs.accessor("other", False) }}
        return out

        {% endfor %}
    {% endfor %}
{% endfor %}