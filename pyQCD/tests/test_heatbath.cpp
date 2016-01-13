#define CATCH_CONFIG_MAIN


#include <algorithms/heatbath.hpp>

#include "helpers.hpp"

TEST_CASE("Heatbath test")
{
  Compare<Real> comp(1.0e-5, 1.0e-8);
  MatrixCompare<SU2Matrix> mat_comp(1.0e-5, 1.0e-8);

  for (unsigned int i = 0; i < 100; ++i) {
    auto heatbath_su2 = pyQCD::gen_heatbath_su2(1.0, 5.0);

    REQUIRE(mat_comp(heatbath_su2 * heatbath_su2.adjoint(),
      SU2Matrix::Identity()));
    auto det = heatbath_su2.determinant();
    REQUIRE(comp(det.real(), 1.0));
    REQUIRE(comp(det.imag(), 0.0));
  }
}