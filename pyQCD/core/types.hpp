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
  typedef double Real;
  typedef std::complex<Real> Complex;
  typedef pyQCD::Lattice<Real> LatticeReal;
  typedef pyQCD::Lattice<Complex> LatticeComplex;
  typedef pyQCD::ColourMatrix<Real, 3> ColourMatrix;
  typedef pyQCD::LatticeColourMatrix<Real, 3> LatticeColourMatrix;
  typedef pyQCD::ColourVector<Real, 3> ColourVector;
  typedef pyQCD::LatticeColourVector<Real, 3> LatticeColourVector;
}
#endif