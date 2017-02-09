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
 * Created by Matt Spraggs on 06/02/17.
 *
 * Tests for the Wilson fermion action.
 */

#define CATCH_CONFIG_MAIN

#include <fermions/wilson_action.hpp>
#include <utils/matrices.hpp>

#include "helpers.hpp"


TEST_CASE ("Testing Wilson fermion action")
{
  typedef pyQCD::ColourMatrix<double, 3> GaugeLink;
  typedef pyQCD::LatticeColourMatrix<double, 3> GaugeField;
  typedef pyQCD::ColourVector<double, 3> SiteFermion;
  typedef pyQCD::LatticeColourVector<double, 3> FermionField;

  pyQCD::Site shape{8, 4, 4, 4};
  pyQCD::LexicoLayout layout(shape);

  GaugeField gauge_field(layout, GaugeLink::Identity(), 4);
  FermionField psi(layout, SiteFermion::Ones(), 4);
  FermionField eta(layout, 4);

  pyQCD::fermions::WilsonAction<double, 3> wilson_action(0.1, gauge_field);

  wilson_action.apply_full(eta, psi);

  MatrixCompare<SiteFermion> comp(1e-5, 1e-8);

  for (unsigned site = 0; site < layout.volume(); ++site) {
    for (unsigned spin = 0; spin < 4; ++spin) {
      REQUIRE (comp(eta(site, spin), SiteFermion::Ones() * 0.1));
    }
  }

  gauge_field.fill(GaugeLink::Zero());
  psi.fill(SiteFermion::Zero());

  pyQCD::Site site{0, 3, 0, 0};
  auto random_mat = pyQCD::random_sun<double, 3>();
  gauge_field(site, 1) = random_mat;
  psi(site, 3) = SiteFermion::Ones();
  site = {0, 0, 0, 1};
  gauge_field(pyQCD::Site{0, 0, 0, 0}, 3) = random_mat;
  psi(site, 2) = SiteFermion::Ones();
  site = {7, 0, 0, 0};
  gauge_field(site, 0) = random_mat;
  psi(site, 2) = SiteFermion::Ones();

  SiteFermion expected =
      -0.5 * (pyQCD::I * (random_mat - random_mat.adjoint())  +
          random_mat.adjoint()) * SiteFermion::Ones();

  wilson_action = pyQCD::fermions::WilsonAction<double, 3>(0.0, gauge_field);

  wilson_action.apply_full(eta, psi);

  REQUIRE (comp(eta[0], expected));
}