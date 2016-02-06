#ifndef TYPES_HPP
#define TYPES_HPP

#include <complex>

#include <Eigen/Dense>

#include "lattice.hpp"

namespace pyQCD {

  template<typename T, int N>
  using ColourMatrix = Eigen::Matrix<std::complex<T>, N, N>;
  template <typename T, int N>
  using ColourVector = Eigen::Matrix<std::complex<T>, N, 1>;
  template <typename T, int N>
  using LatticeColourMatrix
    = pyQCD::Lattice<ColourMatrix<T, N>>;
  template <typename T, int N>
  using LatticeColourVector
    = pyQCD::Lattice<ColourVector<T, N>>;

  template <typename T>
  using SU2Matrix = ColourMatrix<T, 2>;
}

namespace python {
  typedef {{ precision }} Real;
  typedef std::complex<Real> Complex;
  typedef pyQCD::Lattice<Real> LatticeReal;
  typedef pyQCD::Lattice<Complex> LatticeComplex;
  {% for typedef in typedefs %}
  typedef {{ typedef.cpptype }} {{ typedef.cname }};
  {% endfor %}
}
#endif
