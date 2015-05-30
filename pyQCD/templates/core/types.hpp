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
{% for matrix in matrixdefs %}
typedef Eigen::Matrix<Complex, {{ matrix.num_rows }}, {{ matrix.num_cols }}> {{ matrix.matrix_name }};
typedef pyQCD::MatrixArray<{{ matrix.num_rows }}, {{ matrix.num_cols }}, {{ precision }}> {{ matrix.array_name }};
typedef pyQCD::Lattice<Eigen::Matrix<Complex, {{ matrix.num_rows }}, {{ matrix.num_cols }}>, Eigen::aligned_allocator> {{ matrix.lattice_matrix_name }};
typedef pyQCD::Lattice<pyQCD::MatrixArray<{{ matrix.num_rows }}, {{ matrix.num_cols }}, {{ precision }}> > {{ matrix.lattice_array_name }};
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