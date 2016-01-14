/* This file contains the implementations of the functions in heatbath.hpp */

#include "heatbath.hpp"


namespace pyQCD {

  typedef Eigen::Matrix<Complex, 2, 2> Matrix2cd;

  Matrix2cd gen_heatbath_su2(const Real weight, const Real beta)
  {
    // Generate a random SU(2) matrix distributed according to the distribution
    // exp(0.5 * weight * beta * Re tr(X)). We use the algorithm specified in
    // Kennedy and Pendleton (1985), in Phys. Lett. 156B.
    //
    // (See also page 87 of Gattringer and Lang for the source material for
    // this algorithm.)

    typedef Eigen::Matrix<Real, 4, 1> FourVector;

    // Random number generator setup
    // TODO: Implement a proper random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Coefficients for the SU(2) basis of Pauli matrices. This final vector
    // will need to be normalised.
    FourVector coeffs;
    // Now we need to fill the components of this vector. The first component
    // must be distributed according to
    //   sqrt(1 - x^2) * exp(weight * beta * x)
    Real lambda_squared = 2.0;
    Real uniform_squared = std::pow(dis(gen), 2);
    while (uniform_squared > 1 - lambda_squared) {
      Real r0 = 1.0 - dis(gen);
      Real r1 = 1.0 - dis(gen);
      Real r2 = 1.0 - dis(gen);
      lambda_squared
        = - 1.0 / (2.0 * weight * beta)
        * (std::log(r0) + std::pow(std::cos(2 * pi * r1), 2) * std::log(r2));
      uniform_squared = std::pow(dis(gen), 2);
    }
    coeffs[0] = 1 - 2 * lambda_squared;
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

    return construct_su2(coeffs);
  };

  void heatbath_update(LatticeColourMatrix& gauge_field,
                       const LatticeColourMatrix& gauge_action,
                       const Int site_index)
  {
    //auto layout = gauge_field.
  }
}