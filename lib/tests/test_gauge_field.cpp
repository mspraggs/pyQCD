#define CATCH_CONFIG_MAIN

#include <base/gauge_field.hpp>

#include "helpers.hpp"

typedef pyQCD::GaugeField<3> GaugeField;

TEST_CASE("GaugeField test")
{
  GaugeField gauge_field1(4, Eigen::Matrix3cd::Identity());
  GaugeField gauge_field2;
}