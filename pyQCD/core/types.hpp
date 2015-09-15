#ifndef TYPES_HPP
#define TYPES_HPP

#include "array.hpp"
#include "matrix_array.hpp"
#include "lattice.hpp"
#include "constants.hpp"


typedef double Real;
typedef std::complex<Real> Complex;
typedef pyQCD::Array<Complex> ComplexArray;
typedef pyQCD::Lattice<Complex> LatticeComplex;
typedef pyQCD::Lattice<pyQCD::Array<Complex> > LatticeComplexArray;
typedef Eigen::Matrix<Complex, 3, 3> ColourMatrix;
typedef pyQCD::MatrixArray<3, 3, Real> ColourMatrixArray;
typedef pyQCD::Lattice<Eigen::Matrix<Complex, 3, 3>, Eigen::aligned_allocator> LatticeColourMatrix;
typedef pyQCD::Lattice<pyQCD::MatrixArray<3, 3, Real> > GaugeField;
typedef Eigen::Matrix<Complex, 3, 1> ColourVector;
typedef pyQCD::MatrixArray<3, 1, Real> Fermion;
typedef pyQCD::Lattice<Eigen::Matrix<Complex, 3, 1>, Eigen::aligned_allocator> LatticeColourVector;
typedef pyQCD::Lattice<pyQCD::MatrixArray<3, 1, Real> > FermionField;

#endif