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

{% for matrix in matrixdefs %}
{% if matrix.num_cols > 1 %}
inline void mat_assign({{ matrix.matrix_name }}& mat, const int i, const int j, const Complex value)
{ mat(i, j) = value; }

inline void mat_assign({{ matrix.matrix_name }}* mat, const int i, const int j, const Complex value)
{ (*mat)(i, j) = value; }

{% endif %}
{% endfor %}

#endif