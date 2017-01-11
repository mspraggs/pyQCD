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

#define CATCH_CONFIG_MAIN

#include <Eigen/Dense>

#include <core/lattice.hpp>
#include <core/detail/operators.hpp>

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
  std::vector<std::function<double(const double&, const double&)> >
    scalar_operators{
    Plus::apply<double, double>, Minus::apply<double, double>,
    Multiplies::apply<double, double>, Divides::apply<double, double>
  };

  pyQCD::LexicoLayout layout({8, 4, 4, 4});
  TestLayout another_layout({8, 4, 4, 4});

  Lattice lattice1(layout, 1.0, 4);
  Lattice lattice2(layout, 2.0, 4);
  Lattice lattice3(layout, 4u);
  Lattice bad_lattice(another_layout);
  for (unsigned int i = 0; i < bad_lattice.size(); ++i) {
    bad_lattice[i] = i;
  }

  pyQCD::Lattice<Eigen::Matrix3cd> lattice_matrix(
    layout, Eigen::Matrix3cd::Identity() * 4.0);

  SECTION ("Testing array iterators") {
    for (auto elem : lattice1) {
      REQUIRE (elem == 1.0);
    }
  }

  SECTION ("Testing scalar assign") {
    lattice1.fill(4.0);
    for (unsigned int i = 0; i < lattice1.size(); ++i) {
      REQUIRE (lattice1[i] == 4.0);
    }
  }

  SECTION ("Testing array-array binary ops") {
    lattice3 = (lattice1 * 3.0) / 4.0 + lattice2;
    for (unsigned int j = 0; j < lattice1.size(); ++j) {
      REQUIRE (lattice3[j] == (1.0 * 3.0) / 4.0 + 2.0);
    }
  }

  SECTION ("Testing array-scalar binary ops") {
    for (unsigned int i = 0; i < 4; ++i) {
      lattice3 = (lattice1 / 2.0) * 3.0 + 4.0;//lattice_operators[i](lattice1, 2.0);
      for (unsigned int j = 0; j < lattice1.size(); ++j) {
        REQUIRE (lattice3[j] == (1.0 / 2.0) * 3.0 + 4.0);
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
      lattice3 = lattice1;
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
      lattice3 = lattice1;
      lattice_operators[i](lattice3, lattice2);
      for (unsigned int j = 0; j < lattice1.size(); ++j) {
        REQUIRE(lattice3[j] == scalar_operators[i](1.0, 2.0));
      }
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
    decltype(lattice_matrix) result(layout);
    result  = lattice_matrix * (3.0 * Eigen::Matrix3cd::Identity());
    REQUIRE(result.size() == lattice_matrix.size());
    for (auto& site_matrix : result) {
      comparison(site_matrix, Eigen::Matrix3cd::Identity() * 12.0);
    }
  }
}

TEST_CASE("Non-integral Array types test") {
  pyQCD::LexicoLayout layout(std::vector<unsigned int>{8, 4, 4, 4});
  pyQCD::Lattice<Eigen::Matrix3cd> lattice1(layout, Eigen::Matrix3cd::Identity());
  Eigen::Vector3cd vec(1.0, 1.0, 1.0);
  pyQCD::Lattice<Eigen::Vector3cd> vecs(layout);
  vecs = lattice1 * vec;
}
