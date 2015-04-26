#define CATCH_CONFIG_MAIN

#include <base/fermion.hpp>

#include "helpers.hpp"


typedef pyQCD::Fermion<3> Fermion;

TEST_CASE("GaugeField test")
{
  MatrixCompare<Eigen::RowVector3cd> matrix_compare(1.0e-8, 1.0e-5);

  Fermion fermion1(4, Eigen::Vector3cd::Ones());
  Fermion fermion2(4, Eigen::Vector3cd::Random());

  SECTION("Test expressions") {
    Fermion fermion_adj = fermion1.adjoint();
    REQUIRE(matrix_compare(fermion_adj[0], Eigen::RowVector3cd::Ones()));

    fermion_adj = fermion2.adjoint();
    for (int i = 0; i < 4; ++i) {
      REQUIRE(matrix_compare(fermion2[i], fermion_adj[i].adjoint()));
    }
  }
}