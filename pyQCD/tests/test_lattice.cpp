/*
 * This file is part of pyQCD.
 *
 * pyQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * pyQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 *
 * Created by Matt Spraggs on 10/02/16.
 *
 *
 * Tests for the Lattice template class.
 */

#define CATCH_CONFIG_RUNNER

#include <Eigen/Dense>

#include <core/lattice.hpp>

#include "helpers.hpp"

typedef pyQCD::Lattice<double> Lattice;

int partition_arr[] = {2, 2, 1, 1};


TEST_CASE("Lattice test") {
  std::vector<std::function<double(const double&, const double&)> >
    scalar_operators{
    Plus::apply<double, double>, Minus::apply<double, double>,
    Multiplies::apply<double, double>, Divides::apply<double, double>
  };

  pyQCD::Site partition(partition_arr, partition_arr + 4);
  pyQCD::Layout layout({8, 4, 4, 4}, partition, 1, 2);

  Lattice lattice1(layout, 1.0, 4);
  Lattice lattice2(layout, 2.0, 4);

  pyQCD::Lattice<Eigen::Matrix3cd> lattice_matrix(
    layout, Eigen::Matrix3cd::Identity() * 4.0);

  SECTION ("Testing array iterators") {
    for (auto elem : lattice1) {
      REQUIRE (elem == 1.0);
    }
  }

  SECTION ("Testing scalar assign") {
    lattice1 = 4.0;
    for (unsigned int i = 0; i < lattice1.size(); ++i) {
      REQUIRE (lattice1[i] == 4.0);
    }
  }

  SECTION ("Testing array-array binary ops") {
    std::vector<std::function<Lattice(const Lattice&, const Lattice&)> >
      lattice_operators{
      Plus::apply<Lattice, Lattice>, Minus::apply<Lattice, Lattice>,
      Multiplies::apply<Lattice, Lattice>, Divides::apply<Lattice, Lattice>
    };
    for (unsigned int i = 0; i < 4; ++i) {
      Lattice array3 = lattice_operators[i](lattice1, lattice2);
      for (unsigned int j = 0; j < lattice1.size(); ++j) {
        REQUIRE (array3[j] == scalar_operators[i](1.0, 2.0));
      }
    }
  }

  SECTION ("Testing array-scalar binary ops") {
    std::vector<std::function<Lattice(const Lattice&, const double&)> >
      lattice_operators{
      Plus::apply<Lattice, double>, Minus::apply<Lattice, double>,
      Multiplies::apply<Lattice, double>, Divides::apply<Lattice, double>
    };
    for (unsigned int i = 0; i < 4; ++i) {
      Lattice lattice3 = lattice_operators[i](lattice1, 2.0);
      for (unsigned int j = 0; j < lattice1.size(); ++j) {
        REQUIRE (lattice3[j] == scalar_operators[i](1.0, 2.0));
      }
    }
  }

  SECTION ("Testing array-array op-assigns") {
    std::vector<std::function<void(Lattice&, const Lattice&)> >
      lattice_operators{
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 += lattice2; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 -= lattice2; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 *= lattice2; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 /= lattice2; }
    };
    for (unsigned int i = 0; i < 4; ++i) {
      auto lattice3 = lattice1;
      lattice_operators[i](lattice3, lattice2);
      for (unsigned int j = 0; j < lattice1.size(); ++j) {
        REQUIRE(lattice3[j] == scalar_operators[i](1.0, 2.0));
      }
    }
  }

  SECTION ("Testing array-scalar op-assigns") {
    std::vector<std::function<void(Lattice&, const Lattice&)> >
      lattice_operators{
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 += 2.0; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 -= 2.0; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 *= 2.0; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 /= 2.0; }
    };
    for (unsigned int i = 0; i < 4; ++i) {
      auto lattice3 = lattice1;
      lattice_operators[i](lattice3, lattice2);
      for (unsigned int j = 0; j < lattice1.size(); ++j) {
        REQUIRE(lattice3[j] == scalar_operators[i](1.0, 2.0));
      }
    }
  }

  SECTION("Test accessors") {
    lattice1[0] = 500.0;
    REQUIRE(lattice1(80) == 500.0);
    REQUIRE(lattice1(pyQCD::Site{1, 1, 0, 0}) == 500.0);
    lattice1(pyQCD::Site{4, 2, 3, 1}) = 123.0;
    REQUIRE(lattice1(301) == 123.0);
    REQUIRE(lattice1[500] == 123.0);
  }

  SECTION("Test properties") {
    REQUIRE(&lattice1.layout() == &layout);
    REQUIRE(lattice1.size() == 1536);
    REQUIRE(lattice1.num_dims() == 4);
  }

  SECTION("Test non-scalar site types") {
    MatrixCompare<Eigen::Matrix3cd> comparison(1e-5, 1e-8);
    decltype(lattice_matrix) result
      = lattice_matrix * (3.0 * Eigen::Matrix3cd::Identity());
    REQUIRE(result.size() == lattice_matrix.size());
    for (auto& site_matrix : result) {
      comparison(site_matrix, Eigen::Matrix3cd::Identity() * 12.0);
    }
  }

  SECTION("Test lattice views") {
    auto site_view1 = lattice1.site_view(pyQCD::Site{1, 1, 0, 0});
    auto site_view2 = lattice1.site_view(pyQCD::Site{1, 1, 0, 1});
    site_view1[0] = 5.0;
    REQUIRE(lattice1[0] == 5.0);
    site_view1 = site_view2;// + site_view2;
    REQUIRE(lattice1[0] == 1.0);
  }

  SECTION("Test halo swap") {
    auto rank = pyQCD::Communicator::instance().rank();
    if (rank == 0) {
      lattice1(pyQCD::Site{1, 1, 0, 0}) = 144.0;
    }
    lattice1.halo_swap();
    if (rank == 2) {
      REQUIRE(lattice1[1152] == 144.0);
      REQUIRE(lattice1(pyQCD::Site{5, 1, 0, 0}) == 144.0);
    }
    if (rank == 1) {
      REQUIRE(lattice1[896] == 144.0);
      REQUIRE(lattice1(pyQCD::Site{1, 3, 0, 0}) == 144.0);
    }
    if (rank == 3) {
      REQUIRE(lattice1[1472] == 144.0);
      REQUIRE(lattice1(pyQCD::Site{5, 3, 0, 0}) == 144.0);
    }
  }
}

TEST_CASE("Non-integral Array types test") {

  pyQCD::Site partition(partition_arr, partition_arr + 4);
  pyQCD::Layout layout({8, 4, 4, 4}, partition, 1, 2);

  pyQCD::Lattice<Eigen::Matrix3cd> array1(layout, Eigen::Matrix3cd::Identity());
  Eigen::Vector3cd vec(1.0, 1.0, 1.0);
  pyQCD::Lattice<Eigen::Vector3cd> vecs = array1 * vec;
}


int main(int argc, char * argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm;
  int periodic[] = {1, 1, 1, 1};

  MPI_Cart_create(MPI_COMM_WORLD, 4, partition_arr, periodic, 1, &comm);

  pyQCD::Communicator::instance().init(comm);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}