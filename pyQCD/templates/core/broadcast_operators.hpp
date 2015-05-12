#ifndef BROADCAST_OPERATORS_HPP
#define BROADCAST_OPERATORS_HPP

#include "types.hpp"

{% for op in ops %}
auto operator{{ op.1 }}(const {{ op.2.name }}& lhs, const {{ op.3.name }}& rhs)
  -> decltype(lhs {{ op.1 }} rhs)
{
  return lhs{% if op.2.broadcast %}.broadcast(){% endif %} {{ op.1 }} rhs{% if op.3.broadcast %}.broadcast(){% endif %};
}

{% endfor %}
#endif