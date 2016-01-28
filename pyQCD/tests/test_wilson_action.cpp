#define CATCH_CONFIG_MAIN

#include <gauge/wilson_action.hpp>

#include "helpers.hpp"


TEST_CASE("Test Wilson gauge action")
{
  typedef double Real;
  typedef pyQCD::ColourMatrix<Real, 3> ColourMatrix;

  SECTION("Testing computation of staples") {
    MatrixCompare<ColourMatrix> mat_comp(1.0e-8, 1.0e-8);

    pyQCD::LexicoLayout layout({8, 4, 4, 4, 4});
    auto identity = ColourMatrix::Identity();
    pyQCD::LatticeColourMatrix<double, 3> gauge_field(layout, identity);
    pyQCD::Gauge::WilsonAction<double, 3> action(5.0, layout);

    auto staple = action.compute_staples(gauge_field, 0);

    REQUIRE (mat_comp(staple, 6 * identity));

    ColourMatrix rand_mat_1 = ColourMatrix::Random();
    ColourMatrix rand_mat_2 = ColourMatrix::Random();
    ColourMatrix rand_mat_3 = ColourMatrix::Random();
    gauge_field(pyQCD::Site{1, 0, 0, 0, 1}) = rand_mat_1;
    gauge_field(pyQCD::Site{0, 1, 0, 0, 0}) = rand_mat_2;
    gauge_field(pyQCD::Site{0, 0, 0, 0, 1}) = rand_mat_3;

    staple = action.compute_staples(gauge_field, 0);

    ColourMatrix link_product
      = rand_mat_1 * rand_mat_2.adjoint() * rand_mat_3.adjoint();

    REQUIRE (mat_comp(staple, 5.0 * identity + link_product));
  }

  SECTION("Testing local action") {
    Compare<Real> comp(1.0e-8, 1.0e-8);

    pyQCD::LexicoLayout layout({8, 4, 4, 4, 4});
    auto identity = ColourMatrix::Identity();
    pyQCD::LatticeColourMatrix<double, 3> gauge_field(layout, identity);
    pyQCD::Gauge::WilsonAction<double, 3> action(5.0, layout);

    Real local_action = action.local_action(gauge_field, 0);
    REQUIRE(comp(local_action, -30.0));
  }
}
