cimport complex
{% for inc in includes %}
cimport {{ inc.0 }}
{% endfor %}


cdef extern from "types.hpp":
    {% for op in scalar_binary_ops %}
    {{ op.0 }} operator{{ op.1 }}(const {{ op.2 }}&, const {{ op.3 }}&) except +
    {% endfor %}

    {% for op in lattice_binary_ops %}
    {{ op.0 }} operator{{ op.1 }}(const {{ op.2 }}&, const {{ op.3 }}&) except +
    {% endfor %}