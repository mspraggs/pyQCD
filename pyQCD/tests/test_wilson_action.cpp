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
 * Tests for the Wilson gauge action.
 */
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
