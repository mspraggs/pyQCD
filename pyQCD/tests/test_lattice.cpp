/*
 * This file is part of pyQCD.
 *
 * pyQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
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

#define CATCH_CONFIG_MAIN

#include <Eigen/Dense>

#include <core/lattice.hpp>

#include "helpers.hpp"

typedef pyQCD::Lattice<double> Lattice;

class TestLayout : public pyQCD::Layout
{
public:
  TestLayout(const pyQCD::Site& shape) : pyQCD::Layout(shape)
  {
    array_indices_.resize(volume_);
    site_indices_.resize(volume_);
    for (pyQCD::Int i = 0; i < volume_; ++i) {
      array_indices_[i] = volume_ - i - 1;
      site_indices_[volume_ - i - 1] = i;
    }
  }
};

TEST_CASE("Lattice test") {

  pyQCD::LexicoLayout layout({8, 4, 4, 4});
  TestLayout another_layout({8, 4, 4, 4});

  Lattice lattice1(layout, 1.0, 4);
  Lattice lattice2(layout, 2.0, 4);
  Lattice lattice3(layout, 0.0, 4);
  Lattice bad_lattice(another_layout);
  for (unsigned int i = 0; i < bad_lattice.size(); ++i) {
    bad_lattice[i] = i;
  }

  pyQCD::Lattice<Eigen::Matrix3cd> lattice_matrix(
    layout, Eigen::Matrix3cd::Identity() * 4.0);

  SECTION ("Testing scalar assign") {
    lattice1.fill(4.0);
    for (unsigned int i = 0; i < lattice1.size(); ++i) {
      REQUIRE (lattice1[i] == 4.0);
    }
  }

  SECTION ("Testing lattice binary ops") {
    lattice3 = 3.0 * (lattice1 / 2.0 + lattice2) - lattice2;
    for (unsigned int j = 0; j < lattice1.size(); ++j) {
      REQUIRE (lattice3[j] == 3.0 * (1.0 / 2.0 + 2.0) - 2.0);
    }
  }

  SECTION ("Testing lattice-lattice op-assigns") {
    lattice3 = lattice1;
    lattice3 /= lattice2;
    lattice3 += lattice1;
    lattice3 *= lattice2;
    lattice3 -= lattice1;
    for (unsigned int j = 0; j < lattice1.size(); ++j) {
      REQUIRE(lattice3[j] == ((1.0 / 2.0) + 1.0) * 2.0 - 1.0);
    }
  }

  SECTION ("Testing lattice-scalar op-assigns") {
    lattice3 = lattice1;
    lattice3 /= 2.0;
    lattice3 += 1.0;
    lattice3 *= 2.0;
    lattice3 -= 1.0;
    for (unsigned int j = 0; j < lattice1.size(); ++j) {
      REQUIRE(lattice3[j] == ((1.0 / 2.0) + 1.0) * 2.0 - 1.0);
    }
  }

  SECTION("Test accessors") {
    lattice1[0] = 500.0;
    REQUIRE(lattice1(0) == 500.0);
    REQUIRE(lattice1(pyQCD::Site{0, 0, 0, 0}) == 500.0);
    lattice1(pyQCD::Site{4, 2, 3, 1}) = 123.0;
    REQUIRE(lattice1(301) == 123.0);
    REQUIRE(lattice1[1204] == 123.0);
  }

  SECTION("Test properties") {
    REQUIRE(&lattice1.layout() == &layout);
    REQUIRE(lattice1.size() == 2048);
    REQUIRE(lattice1.num_dims() == 4);
  }

  SECTION("Test non-scalar site types") {
    MatrixCompare<Eigen::Matrix3cd> comparison(1e-5, 1e-8);
    pyQCD::Lattice<Eigen::Matrix3cd> result(lattice_matrix.layout());
    result = lattice_matrix * (3.0 * Eigen::Matrix3cd::Identity());
    REQUIRE(result.size() == lattice_matrix.size());
    for (unsigned int i = 0; i < lattice_matrix.size(); ++i) {
      comparison(lattice_matrix[i], Eigen::Matrix3cd::Identity() * 12.0);
    }
  }

  SECTION("Test change of layout") {

    for (unsigned int i = 0; i < lattice1.size(); ++i) {
      lattice1[i] = static_cast<double>(i);
    }

    lattice1.change_layout(another_layout);

    for (unsigned int i = 0; i < lattice1.volume(); ++i) {
      for (unsigned int j = 0; j < lattice1.site_size(); ++j) {
        auto expected =
            static_cast<double>(4 * (lattice1.volume() - i - 1) + j);
        REQUIRE(lattice1[4 * i + j] == expected);
      }
    }
  }

  SECTION("Test even and odd site views") {

    for (unsigned int i = 0; i < lattice1.size(); ++i) {
      lattice1[i] = static_cast<double>(i);
    }

    auto even_view = lattice1.even_sites_view();
    auto odd_view = lattice1.odd_sites_view();

    REQUIRE(even_view[0] == 0.0);
    REQUIRE(even_view[4] == 8.0);
    REQUIRE(even_view[8] == 20.0);
    REQUIRE(even_view[12] == 28.0);

    REQUIRE(odd_view[0] == 4.0);
    REQUIRE(odd_view[4] == 12.0);
    REQUIRE(odd_view[8] == 16.0);
    REQUIRE(odd_view[12] == 24.0);

    lattice3.odd_sites_view() = odd_view - lattice2.odd_sites_view();

    REQUIRE(lattice3[4] == 4.0 - 2.0);
  }
}

TEST_CASE("Non-integral Lattice types test") {
  pyQCD::LexicoLayout layout(std::vector<unsigned int>{8, 4, 4, 4});
  pyQCD::Lattice<double> lattice_double(layout, 5.0);
  pyQCD::Lattice<Eigen::Matrix3cd> lattice_matrix(
    layout, Eigen::Matrix3cd::Identity());
  Eigen::Vector3cd vec(1.0, 1.0, 1.0);
  pyQCD::Lattice<Eigen::Vector3cd> vecs(lattice_matrix.layout());
  vecs = lattice_matrix * vec * lattice_double;
}
