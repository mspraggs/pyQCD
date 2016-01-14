#define CATCH_CONFIG_MAIN


#include <algorithms/heatbath.hpp>

#include "helpers.hpp"

TEST_CASE("Heatbath test")
{
  Compare<Real> comp(1.0e-5, 1.0e-8);
  MatrixCompare<SU2Matrix> mat_comp(1.0e-5, 1.0e-8);

  const unsigned int n = 10000;
  std::vector<Real> x0s(n);
  for (unsigned int i = 0; i < n; ++i) {
    auto heatbath_su2 = pyQCD::gen_heatbath_su2(1.0, 5.0);

    REQUIRE(mat_comp(heatbath_su2 * heatbath_su2.adjoint(),
      SU2Matrix::Identity()));
    auto det = heatbath_su2.determinant();
    REQUIRE(comp(det.real(), 1.0));
    REQUIRE(comp(det.imag(), 0.0));

    x0s[i] = heatbath_su2.trace().real() / 2.0;
  }
  // Compute the mean and the standard deviation of x0 (coefficient on sigma0).
  Real mean = std::accumulate(x0s.begin(), x0s.end(), 0.0) / n;

  std::vector<Real> square_devs(n);
  std::transform(x0s.begin(), x0s.end(), square_devs.begin(),
                 [mean] (const Real val)
                 { return (val - mean) * (val - mean); });
  Real sum_square_devs
    = std::accumulate(square_devs.begin(), square_devs.end(), 0.0);
  Real stddev = std::sqrt(sum_square_devs / n);

  Compare<Real> comp_weak(0.005, 0.005);
  REQUIRE(comp_weak(mean, 0.7193405813643129));
  REQUIRE(comp_weak(stddev, 0.2257095017580442));
}