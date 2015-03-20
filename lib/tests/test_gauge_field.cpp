#define CATCH_CONFIG_MAIN

#include <base/gauge_field.hpp>

#include "helpers.hpp"


typedef pyQCD::GaugeField<3> GaugeField;

MatrixCompare<Eigen::Matrix3cd> matrix_compare(1.0e-8, 1.0e-5);

TEST_CASE("GaugeField test")
{
  GaugeField gauge_field1(4, Eigen::Matrix3cd::Identity());
  GaugeField gauge_field2;

  SECTION("Test expressions") {
    GaugeField gauge_field_adj = gauge_field1.adjoint();
    REQUIRE(matrix_compare(gauge_field_adj[0], Eigen::Matrix3cd::Identity()));
  }
}