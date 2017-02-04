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
 * Created by Matt Spraggs on 02/02/17.
 *
 * Tests for 4D hopping matrix
 */

#define CATCH_CONFIG_MAIN

#include <fermions/hopping_matrix.hpp>
#include <utils/random.hpp>

#include "helpers.hpp"

TEST_CASE ("Testing hopping matrix")
{
  typedef pyQCD::ColourVector<double, 3> SiteFermion;
  typedef pyQCD::LatticeColourVector<double, 3> LatticeFermion;
  typedef pyQCD::ColourMatrix<double, 3> GaugeLink;
  typedef pyQCD::LatticeColourMatrix<double, 3> GaugeField;

  pyQCD::LexicoLayout layout({8, 4, 4, 4});

  double random_float = pyQCD::rng().generate_real(0.0, 1.0);

  GaugeField gauge_field(layout, GaugeLink::Identity() * random_float, 4);
  LatticeFermion fermion_in(layout, SiteFermion::Ones(), 4);
  LatticeFermion fermion_out(layout, 4);

  auto identity = Eigen::Matrix4cd::Identity();

  pyQCD::aligned_vector<Eigen::Matrix4cd> spin_structures(8, identity);

  auto hopping_matrix =
      pyQCD::fermions::HoppingMatrix<double, 3, 1>(gauge_field,
                                                   spin_structures);

  hopping_matrix.apply_full(fermion_out, fermion_in);

  Compare<double> comp;

  for (unsigned a = 0; a < 3; ++a) {
    REQUIRE(comp(fermion_out[0][a].real(), 8 * random_float));
    REQUIRE(comp(fermion_out[0][a].imag(), 0.0));
  }
}