#ifndef MATRIX_FUNCS_HPP
#define MATRIX_FUNCS_HPP

/* This file contains utility function relating to matrices. */

#include <core/types.hpp>
#include <utils/math.hpp>


namespace pyQCD {

  // Define Pauli matrices
  typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> DynamicSU2;
  const SU2Matrix sigma0 = SU2Matrix::Identity();
  const SU2Matrix sigma1 = (DynamicSU2(2, 2) << 0.0, 1.0, 1.0, 0.0).finished();
  const SU2Matrix sigma2 = (DynamicSU2(2, 2) << 0.0, -I, I, 0.0).finished();
  const SU2Matrix sigma3 = (DynamicSU2(2, 2) << 1.0, 0.0, 0.0, -1.0).finished();

  template <typename T>
  SU2Matrix construct_su2(const T& coefficients)
  {
    // This function constructs an SU(2) matrix from a 4-vector and the above
    // Pauli matrices.
    return coefficients[0] * sigma0
           + I * (coefficients[1] * sigma1
                  + coefficients[2] * sigma2
                  + coefficients[3] * sigma3);
  }

  void compute_su2_subgroup_pos(const unsigned int index,
                                unsigned int& i, unsigned int& j);

  SU2Matrix extract_su2(const ColourMatrix& colour_matrix,
                        const unsigned int subgroup);

  ColourMatrix insert_su2(const SU2Matrix& su2_matrix,
                          const unsigned int subgroup);
}

#endif