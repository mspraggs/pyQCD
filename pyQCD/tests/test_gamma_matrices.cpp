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

  auto gamma_matrices = pyQCD::generate_gamma_matrices<double>(2);

  REQUIRE (gamma_matrices.size() == 2);
  REQUIRE (comp(gamma_matrices[0], pyQCD::sigma1));
  REQUIRE (comp(gamma_matrices[1], pyQCD::sigma2));

  for (int d = 4; d < 12; d += 2) {
    gamma_matrices = pyQCD::generate_gamma_matrices<double>(d);

    REQUIRE (gamma_matrices.size() == d);

    for (int i = 0; i < d; ++i) {
      if (!comp(gamma_matrices[i], gamma_matrices[i].adjoint())) {
        std::cout << d << ", " << i << std::endl;
        std::cout << gamma_matrices[i] << std::endl;
      }
      REQUIRE (comp(gamma_matrices[i], gamma_matrices[i].adjoint()));
    }
  }
}