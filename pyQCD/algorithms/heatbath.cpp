/* This file contains the implementations of the functions in heatbath.hpp */

#include "heatbath.hpp"


namespace pyQCD {

  typedef Eigen::Matrix<Complex, 2, 2> Matrix2cd;

  Matrix2cd gen_heatbath_su2(const Real weight, const Real beta)
  {
    // Generate a random SU(2) matrix distributed according to the requirements
    // of the heatbath algorithm. (See page 87 of Gattringer and Lang for the
    // source material for this algorithm.)

    typedef Eigen::Matrix<Real, 4, 1> FourVector;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    FourVector coeffs;

    Real lambda_squared = 2.0;
    Real uniform_squared = std::pow(dis(gen), 2);
    while (uniform_squared > 1 - lambda_squared) {
      Real r0 = 1.0 - dis(gen);
      Real r1 = 1.0 - dis(gen);
      Real r2 = 1.0 - dis(gen);
      lambda_squared
        = - 1.0 / (2.0 * weight * beta)
        * (std::log(r0)
           + std::pow(std::cos(2 * pi * r1), 2) * std::log(r2));
      uniform_squared = std::pow(dis(gen), 2);
    }

    coeffs[0] = 1 - 2 * lambda_squared;
    Real three_vec_magnitude = std::sqrt(1 - coeffs[0] * coeffs[0]);

    Real cos_theta = 2.0 * dis(gen) - 1.0;
    Real sin_theta = std::sqrt(1 - cos_theta * cos_theta);
    Real phi = 2 * pi * dis(gen);

    coeffs[1] = three_vec_magnitude * sin_theta * std::cos(phi);
    coeffs[2] = three_vec_magnitude * sin_theta * std::sin(phi);
    coeffs[3] = three_vec_magnitude * cos_theta;

    return construct_su2(coeffs);
  };

  void heatbath_update(LatticeColourMatrix& gauge_field,
                       const LatticeColourMatrix& gauge_action,
                       const Int site_index)
  {
    //auto layout = gauge_field.
  }
}