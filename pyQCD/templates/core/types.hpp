#ifndef TYPES_HPP
#define TYPES_HPP

#include "array.hpp"
#include "matrix_array.hpp"
#include "lattice.hpp"
#include "constants.hpp"


typedef {{ precision }} Real;
typedef std::complex<Real> Complex;
typedef pyQCD::Array<Complex> ComplexArray;
typedef pyQCD::Lattice<Complex> LatticeComplex;
typedef pyQCD::Lattice<pyQCD::Array<Complex> > LatticeComplexArray;
{% for typedef in typedefs %}
typedef {{ typedef|cpptype }} {{ typedef.cname }};
{% endfor %}

#endif
