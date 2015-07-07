#ifndef TYPES_HPP
#define TYPES_HPP

#include "array.hpp"
#include "matrix_array.hpp"
#include "lattice.hpp"


typedef {{ precision }} Real;
typedef std::complex<{{ precision }}> Complex;
typedef pyQCD::Array<Complex> ComplexArray;
typedef pyQCD::Lattice<Complex> LatticeComplex;
typedef pyQCD::Lattice<pyQCD::Array<Complex> > LatticeComplexArray;
{% for typedef in typedefs %}
typedef {{ typedef|cpptype(precision) }} {{ typedef.cname }};
{% endfor %}

{% for typedef in typedefs %}
{% if typedef.structure[0] == "Matrix" %}
{% if typedef.is_matrix %}
inline void mat_assign({{ typedef.cname }}& mat, const int i, const int j, const Complex value)
{ mat(i, j) = value; }

inline void mat_assign({{ typedef.cname }}* mat, const int i, const int j, const Complex value)
{ (*mat)(i, j) = value; }

{% endif %}
{% endif %}
{% endfor %}

#endif