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
#include <utils/matrices.hpp>
#include <utils/random.hpp>

#include "helpers.hpp"

TEST_CASE ("Testing hopping matrix")
{
  typedef pyQCD::ColourVector<double, 3> SiteFermion;
  typedef pyQCD::LatticeColourVector<double, 3> LatticeFermion;
  typedef pyQCD::ColourMatrix<double, 3> GaugeLink;
  typedef pyQCD::LatticeColourMatrix<double, 3> GaugeField;

  pyQCD::LexicoLayout layout({8, 4, 4, 4});

  GaugeField gauge_field(layout, GaugeLink::Identity(), 4);
  LatticeFermion fermion_in(layout, SiteFermion::Ones(), 4);
  LatticeFermion fermion_out(layout, 4);

  SiteFermion expected_result = SiteFermion::Zero();
  for (unsigned d = 0; d < 4; ++d) {
    auto random_mat = pyQCD::random_sun<double, 3>();
    double random_double = pyQCD::rng().generate_real(0.0, 1.0);
    pyQCD::Site site{0, 0, 0, 0};
    gauge_field(site, d) = random_mat;
    site[d] = 1;
    fermion_in(site, 0) *= random_double;
    expected_result += random_double * random_mat * SiteFermion::Ones();

    random_mat = pyQCD::random_sun<double, 3>();
    random_double = pyQCD::rng().generate_real(0.0, 1.0);
    site[d] = layout.shape()[d] - 1;
    gauge_field(site, d) = random_mat;
    fermion_in(site, 0) *= random_double;
    expected_result +=
        random_double * random_mat.adjoint() * SiteFermion::Ones();
  }

  auto identity = Eigen::Matrix4cd::Identity();

  std::vector<Eigen::MatrixXcd> spin_structures(8, identity);

  std::vector<std::complex<double>> boundary_phases(4, 1.0);

  auto hopping_matrix =
      pyQCD::fermions::HoppingMatrix<double, 3, 1>(gauge_field, boundary_phases,
                                                   spin_structures);

  hopping_matrix.apply_full(fermion_out, fermion_in);

  MatrixCompare<SiteFermion> comp(1e-5, 1e-8);

  REQUIRE(comp(fermion_out[0], expected_result));
}