#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

/* Constants for the C++ code live here. Mainly just the number of colours. */

{% for type, name, value in constants %}
{{ type }} {{ name }} = {{ value }};
{% endfor %}

#endif