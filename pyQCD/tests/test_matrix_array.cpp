#define CATCH_CONFIG_MAIN

#include <core/matrix_array.hpp>

#include "helpers.hpp"


typedef pyQCD::MatrixArray<3, 1> Fermion;
typedef pyQCD::MatrixArray<3, 3> GaugeLinks;

TEST_CASE("GaugeField test")
{
  MatrixCompare<Eigen::RowVector3cd> fermion_compare(1.0e-8, 1.0e-5);
  MatrixCompare<Eigen::Matrix3cd> links_compare(1.0e-8, 1.0e-5);

  Fermion fermion1(4, Eigen::Vector3cd::Ones());
  Fermion fermion2(4, Eigen::Vector3cd::Random());
  GaugeLinks links1(4, Eigen::Matrix3cd::Ones());
  GaugeLinks links2(4, Eigen::Matrix3cd::Random());

  SECTION("Test expressions") {
    Fermion fermion_adj = fermion1.adjoint();
    REQUIRE(fermion_compare(fermion_adj[0], Eigen::RowVector3cd::Ones()));

    fermion_adj = fermion2.adjoint();
    for (int i = 0; i < 4; ++i) {
      REQUIRE(fermion_compare(fermion2[i], fermion_adj[i].adjoint()));
    }

    GaugeLinks links_adj = links1.adjoint();
    REQUIRE(links_compare(links_adj[0], Eigen::Matrix3cd::Ones()));

    links_adj = links2.adjoint();
    for (int i = 0; i < 4; ++i) {
      REQUIRE(links_compare(links2[i], links_adj[i].adjoint()));
    }
  }
}
