#ifndef TYPES_HPP
#define TYPES_HPP

#include <complex>

#include <Eigen/Dense>

#include "lattice.hpp"
#include "constants.hpp"


typedef {{ precision }} Real;
typedef std::complex<Real> Complex;
typedef pyQCD::Lattice<Complex> LatticeComplex;
{% for typedef in typedefs %}
typedef {{ typedef|cpptype }} {{ typedef.cname }};
{% endfor %}

#endif
