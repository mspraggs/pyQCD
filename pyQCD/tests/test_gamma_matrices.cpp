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
 * Created by Matt Spraggs on 05/02/17.
 *
 * Test of gamma matrix generation
 */

#include <utils/matrices.hpp>

#include "helpers.hpp"


TEST_CASE ("Testing gamma matrix generation")
{
  MatrixCompare<Eigen::MatrixXcd> comp(1e-5, 1e-8);

  const auto gamma_matrices_2 = pyQCD::generate_gamma_matrices<double>(2);

  REQUIRE (gamma_matrices_2.size() == 2);
  REQUIRE (comp(gamma_matrices_2[0], pyQCD::sigma1));
  REQUIRE (comp(gamma_matrices_2[1], pyQCD::sigma2));

  for (int d = 4; d < 12; d += 2) {
    const auto gamma_matrices_d = pyQCD::generate_gamma_matrices<double>(d);

    REQUIRE (gamma_matrices_d.size() == d);

    for (int i = 0; i < d; ++i) {
      REQUIRE (comp(gamma_matrices_d[i], gamma_matrices_d[i].adjoint()));
    }
  }
}