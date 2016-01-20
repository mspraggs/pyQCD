#ifndef MATRIX_FUNCS_HPP
#define MATRIX_FUNCS_HPP

/* This file contains utility function relating to matrices. */

#include <core/types.hpp>
#include <utils/math.hpp>


namespace pyQCD {

  // Define Pauli matrices
  const SU2Matrix<double> sigma0 = SU2Matrix<double>::Identity();
  const SU2Matrix<double> sigma1
    = (Eigen::MatrixXcd(2, 2) << 0.0, 1.0, 1.0, 0.0).finished();
  const SU2Matrix<double> sigma2
    = (Eigen::MatrixXcd(2, 2) << 0.0, -I, I, 0.0).finished();
  const SU2Matrix<double> sigma3
    = (Eigen::MatrixXcd(2, 2) << 1.0, 0.0, 0.0, -1.0).finished();

  template <typename Real, typename U>
  SU2Matrix<Real> construct_su2(const U& coefficients)
  {
    // This function constructs an SU(2) matrix from a 4-vector and the above
    // Pauli matrices.
    return coefficients[0] * sigma0
           + I * (coefficients[1] * sigma1
                  + coefficients[2] * sigma2
                  + coefficients[3] * sigma3);
  }

  template <typename Real>
  SU2Matrix<Real> random_su2()
  {
    // Generate a random SU(2) matrix using the Pauli basis.
    // TODO: Implement a proper random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Basically we want to create a random normalised 4-vector from a
    // hyper-spherically symmetric distribution.
    Real coeffs[4];
    coeffs[0] = dis(gen);
    // With the first component determined, the magnitude of the remaining
    // three-vector can easily be determined.
    Real three_vec_magnitude = std::sqrt(1 - coeffs[0] * coeffs[0]);
    // The remaining three-vector should then be take from a uniform spherical
    // distribution.
    Real cos_theta = 2.0 * dis(gen) - 1.0;
    Real sin_theta = std::sqrt(1 - cos_theta * cos_theta);
    Real phi = 2 * pi * dis(gen);

    coeffs[1] = three_vec_magnitude * sin_theta * std::cos(phi);
    coeffs[2] = three_vec_magnitude * sin_theta * std::sin(phi);
    coeffs[3] = three_vec_magnitude * cos_theta;

    return construct_su2<Real>(coeffs);
  }

  template <int Nc>
  void compute_su2_subgroup_pos(const unsigned int index,
                                unsigned int& i, unsigned int& j)
  {
    pyQCDassert((index < Nc), std::range_error("SU(2) subgroup index invalid"));

    unsigned int tmp = index;
    for (i = 0; tmp >= 3 - 1 - i; ++i) {
      tmp -= (Nc - 1 - i);
    }
    j = i + 1 + tmp;
  }

  template <typename Real, int Nc>
  SU2Matrix<Real> extract_su2(const ColourMatrix<Real, Nc> colour_matrix,
                              const unsigned int subgroup)
  {
    typedef SU2Matrix<Real> Mat;
    Mat ret;

    unsigned int i, j;
    compute_su2_subgroup_pos<Nc>(subgroup, i, j);

    ret(0, 0) = colour_matrix(i, i);
    ret(0, 1) = colour_matrix(i, j);
    ret(1, 0) = colour_matrix(j, i);
    ret(1, 1) = colour_matrix(j, j);

    return ret - ret.adjoint() + Mat::Identity() * std::conj(ret.trace());
  }

  template <int Nc, typename Real>
  ColourMatrix<Real, Nc> insert_su2(const SU2Matrix<Real>& su2_matrix,
                                    const unsigned int subgroup)
  {
    ColourMatrix<Real, Nc> ret = ColourMatrix<Real, Nc>::Identity();
    unsigned int i, j;
    compute_su2_subgroup_pos<Nc>(subgroup, i, j);

    ret(i, i) = su2_matrix(0, 0);
    ret(i, j) = su2_matrix(0, 1);
    ret(j, i) = su2_matrix(1, 0);
    ret(j, j) = su2_matrix(1, 1);

    return ret;
  }
}

#endif