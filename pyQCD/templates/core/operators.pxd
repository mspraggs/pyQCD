
cdef extern from "types.hpp":
    {% for op in scalar_binary_ops %}
    {{ op.0 }} operator{{ op.1 }}(const {{ op.2 }}&, const {{ op.3 }}&) except +
    {% endfor %}

    {% for op in non_broadcast_binary_ops %}
    {{ op.0 }} operator{{ op.1 }}(const {{ op.2.name }}&, const {{ op.3.name }}&) except +
    {% endfor %}

cdef extern from "broadcast_operators.hpp":
    {% for op in broadcast_binary_ops %}
    {{ op.0 }} operator{{ op.1 }}(const {{ op.2.name }}&, const {{ op.3.name}}&) except +
    {% endfor %}