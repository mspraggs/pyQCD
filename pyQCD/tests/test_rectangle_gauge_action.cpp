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
 * Created by Matt Spraggs on 29/01/2017.
 *
 *
 * Tests for the Wilson gauge action.
 */

#include <gauge/rectangle_action.hpp>

#include "helpers.hpp"


TEST_CASE("Test rectangle gauge action")
{
  using Real = double;
  using ColourMatrix = pyQCD::ColourMatrix<Real, 3>;

  SECTION("Testing computation of staples") {
    const MatrixCompare<ColourMatrix> mat_comp(1.0e-8, 1.0e-8);

    const pyQCD::LexicoLayout layout({8, 4, 4, 4});
    const auto identity = ColourMatrix::Identity();
    pyQCD::LatticeColourMatrix<double, 3> gauge_field(layout, identity, 4);
    const pyQCD::gauge::RectangleAction<double, 3> action(5.0, layout, 1.0);

    auto staple = action.compute_staples(gauge_field, 0);

    REQUIRE (mat_comp(staple, -24 * identity));

    for (pyQCD::Int d = 1; d < 4; ++d) {
      const ColourMatrix rand_mat_1 = ColourMatrix::Random();
      const ColourMatrix rand_mat_2 = ColourMatrix::Random();
      const ColourMatrix rand_mat_3 = ColourMatrix::Random();

      const pyQCD::Site site1{1, 0, 0, 0};
      auto site2 = site1;
      auto site3 = site1;
      site2[0] = 0;
      site2[d] = 1;
      site3[0] = 0;
      site3[d] = 0;

      gauge_field(site1, d) = rand_mat_1;
      gauge_field(site2, 0) = rand_mat_2;
      gauge_field(site3, d) = rand_mat_3;

      staple = action.compute_staples(gauge_field, 0);

      gauge_field(site1, d) = ColourMatrix::Identity();
      gauge_field(site2, 0) = ColourMatrix::Identity();
      gauge_field(site3, d) = ColourMatrix::Identity();

      const ColourMatrix plaquette_link_product
          = rand_mat_1 * rand_mat_2.adjoint() * rand_mat_3.adjoint();
      const ColourMatrix rectangle_link_product_1 =
          rand_mat_1 * rand_mat_2.adjoint();
      const ColourMatrix rectangle_link_product_2 =
          rand_mat_1 * rand_mat_3.adjoint();
      const ColourMatrix rectangle_link_product_3
          = rand_mat_2.adjoint() * rand_mat_3.adjoint();

      const ColourMatrix plane_0_staples
          = -7.0 * plaquette_link_product + rectangle_link_product_1
              + rectangle_link_product_2 + rectangle_link_product_3
              - 4.0 * identity;

      const ColourMatrix expected_mat = plane_0_staples - 16.0 * identity;

      REQUIRE (mat_comp(staple, expected_mat));
    }
  }

  SECTION("Testing local action") {
    const Compare<Real> comp(1.0e-8, 1.0e-8);

    const pyQCD::LexicoLayout layout({8, 4, 4, 4});
    const auto identity = ColourMatrix::Identity();
    const pyQCD::LatticeColourMatrix<double, 3> gauge_field(layout, identity, 4);
    const pyQCD::gauge::RectangleAction<double, 3> action(5.0, layout, 1.0);

    const Real local_action = action.local_action(gauge_field, 0);
    REQUIRE(comp(local_action, 120.0));
  }
}
