cimport complex
{% for typedef in typedefs %}
cimport {{ typedef.cmodule }}
{% endfor %}


cdef extern from "types.hpp":
{% for op in operations %}
{% set opoperations = operations[op] %}
{% for ret, lhs, rhs in opoperations %}
    {{ ret.cmodule }}.{{ ret.cname }} operator{{ op }}(const {% if not lhs.builtin %}{{ lhs.cmodule }}.{% endif %}{{ lhs.cname }}{% if not lhs.builtin %}&{% endif %}, const {% if not rhs.builtin %}{{ rhs.cmodule }}.{% endif %}{{ rhs.cname }}{% if not rhs.builtin %}&{% endif %})
{% endfor %}
{% endfor %}