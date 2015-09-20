#ifndef TYPES_HPP
#define TYPES_HPP

#include <complex>

#include <Eigen/Dense>

#include "lattice.hpp"
#include "constants.hpp"


typedef double Real;
typedef std::complex<Real> Complex;
typedef pyQCD::Lattice<Complex> LatticeComplex;
typedef Eigen::Matrix<Complex, 3, 3> ColourMatrix;
typedef pyQCD::Lattice<Eigen::Matrix<Complex, 3, 3>, Eigen::aligned_allocator> LatticeColourMatrix;
typedef Eigen::Matrix<Complex, 3, 1> ColourVector;
typedef pyQCD::Lattice<Eigen::Matrix<Complex, 3, 1>, Eigen::aligned_allocator> LatticeColourVector;

#endif