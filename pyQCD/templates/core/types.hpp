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
{% for matrix in matrices %}
typedef Eigen::Matrix<Complex, {{ matrix.num_rows }}, {{ matrix.num_cols }}> {{ matrix.matrix_name }};
typedef pyQCD::MatrixArray<{{ matrix.num_rows }}, {{ matrix.num_cols }}, {{ precision }}> {{ matrix.array_name }};
typedef pyQCD::Lattice<Eigen::Matrix<Complex, {{ matrix.num_rows }}, {{ matrix.num_cols }}>, Eigen::aligned_allocator> {{ matrix.lattice_matrix_name }};
typedef pyQCD::Lattice<pyQCD::MatrixArray<{{ matrix.num_rows }}, {{ matrix.num_cols }}, {{ precision }}> > {{ matrix.lattice_array_name }};
{% endfor %}

#endif