#define CATCH_CONFIG_MAIN

#include <base/gauge_field.hpp>

#include "helpers.hpp"


typedef pyQCD::GaugeField<3> GaugeField;

TEST_CASE("GaugeField test")
{
  MatrixCompare<Eigen::Matrix3cd> matrix_compare(1.0e-8, 1.0e-5);

  GaugeField gauge_field1(4, Eigen::Matrix3cd::Identity());
  GaugeField gauge_field2(4, Eigen::Matrix3cd::Random());

  SECTION("Test expressions") {
    GaugeField gauge_field_adj = gauge_field1.adjoint();
    REQUIRE(matrix_compare(gauge_field_adj[0], Eigen::Matrix3cd::Identity()));

    gauge_field_adj = gauge_field2.adjoint();
    for (int i = 0; i < 4; ++i) {
      REQUIRE(matrix_compare(gauge_field2[i], gauge_field_adj[i].adjoint()));
    }
  }
}