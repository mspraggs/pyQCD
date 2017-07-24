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
 * Created by Matt Spraggs on 12/02/17.
 *
 * Test of the conjugate gradient algorithm.
 */

#include <algorithms/conjugate_gradient.hpp>
#include <fermions/wilson_action.hpp>

#include "helpers.hpp"


template <typename Real, int Nc>
class TestFermionAction : public pyQCD::fermions::Action<Real, Nc>
{
public:
  TestFermionAction(const Real mass, const int ndims)
    : pyQCD::fermions::Action<Real, Nc>(mass, std::vector<Real>(ndims, 1.0))
  {}

  void apply_full(pyQCD::LatticeColourVector<Real, Nc>& fermion_out,
                  const pyQCD::LatticeColourVector<Real, Nc>& fermion_in) const
  { fermion_out = this->mass_ * fermion_in; }

  virtual void apply_even_even_inv(
      pyQCD::LatticeColourVector<Real, Nc>& fermion_out,
      const pyQCD::LatticeColourVector<Real, Nc>& fermion_in) const {}
  virtual void apply_odd_odd(
      pyQCD::LatticeColourVector<Real, Nc>& fermion_out,
      const pyQCD::LatticeColourVector<Real, Nc>& fermion_in) const {}
  virtual void apply_even_odd(
      pyQCD::LatticeColourVector<Real, Nc>& fermion_out,
      const pyQCD::LatticeColourVector<Real, Nc>& fermion_in) const {}
  virtual void apply_odd_even(
      pyQCD::LatticeColourVector<Real, Nc>& fermion_out,
      const pyQCD::LatticeColourVector<Real, Nc>& fermion_in) const {}

  void apply_hermiticity(pyQCD::LatticeColourVector<Real, Nc>& fermion) const
  { }
  void remove_hermiticity(pyQCD::LatticeColourVector<Real, Nc>& fermion) const
  { }
};


TEST_CASE ("Test of unpreconditioned conjugate gradient algorithm")
{
  typedef pyQCD::ColourVector<double, 3> SiteFermion;
  typedef pyQCD::LatticeColourVector<double, 3> LatticeFermion;

  pyQCD::LexicoLayout layout({8, 4, 4, 4});

  LatticeFermion src(layout, SiteFermion::Zero(), 4);
  src[0][0] = 1.0;

  SECTION ("Testing simple proportional action")
  {
    TestFermionAction<double, 3> action(2.0, 4);

    auto result = pyQCD::conjugate_gradient_unprec(action, src, 1000, 1e-10);

    for (int i = 0; i < 3; ++i) {
      REQUIRE (result.solution()[0][i].real() == (i == 0 ? 0.5 : 0.0));
      REQUIRE (result.solution()[0][i].imag() == 0.0);
    }

    REQUIRE (result.tolerance() == 0);
    REQUIRE (result.num_iterations() == 1);
  }

  SECTION ("Testing Wilson action")
  {
    typedef pyQCD::ColourMatrix<double, 3> GaugeLink;
    typedef pyQCD::LatticeColourMatrix<double, 3> GaugeField;

    GaugeField gauge_field(layout, GaugeLink::Identity(), 4);

    std::vector<double> boundary_rotations(4, 1.0);

    pyQCD::fermions::WilsonAction<double, 3> action(0.1, gauge_field,
                                                    boundary_rotations);

    auto result = pyQCD::conjugate_gradient_unprec(action, src, 1000, 1e-8);

    MatrixCompare<SiteFermion> compare(1e-7, 1e-9);
    SiteFermion expected = SiteFermion::Zero();
    expected[0] = std::complex<double>(0.2522536470229704,
                                       1.1333971980249629e-13);

    REQUIRE (compare(result.solution()[0], expected));
    REQUIRE ((result.tolerance() < 1e-8 && result.tolerance() > 0));
    REQUIRE (result.num_iterations() == 69);

    LatticeFermion lhs(layout, 4);
    action.apply_full(lhs, result.solution());

    for (unsigned int i = 0; i < lhs.size(); ++i) {
      REQUIRE (compare(lhs[i], src[i]));
    }
  }
}


TEST_CASE("Testing even-odd preconditioned conjugate gradient algorithm")
{
  typedef pyQCD::ColourVector<double, 3> SiteFermion;
  typedef pyQCD::LatticeColourVector<double, 3> LatticeFermion;

  pyQCD::LexicoLayout lexico_layout({8, 4, 4, 4});
  pyQCD::EvenOddLayout even_odd_layout({8, 4, 4, 4});

  LatticeFermion src(lexico_layout, SiteFermion::Zero(), 4);
  src[0][0] = 1.0;
  src.change_layout(even_odd_layout);

  SECTION ("Testing Wilson action")
  {
    typedef pyQCD::ColourMatrix<double, 3> GaugeLink;
    typedef pyQCD::LatticeColourMatrix<double, 3> GaugeField;

    GaugeField gauge_field(even_odd_layout, GaugeLink::Identity(), 4);

    std::vector<double> boundary_rotations(4, 1.0);

    pyQCD::fermions::WilsonAction<double, 3> action(0.1, gauge_field,
                                                    boundary_rotations);

    auto result = pyQCD::conjugate_gradient_eoprec(action, src, 1000, 1e-8);

    MatrixCompare<SiteFermion> compare(1e-7, 1e-9);
    SiteFermion expected = SiteFermion::Zero();
    expected[0] = std::complex<double>(0.2522536470229704,
                                       1.1333971980249629e-13);

    REQUIRE (compare(result.solution()[0], expected));
    REQUIRE ((result.tolerance() < 1e-8 && result.tolerance() > 0));
    REQUIRE (result.num_iterations() == 29);

    LatticeFermion lhs(even_odd_layout, 4);
    action.apply_full(lhs, result.solution());

    for (unsigned int i = 0; i < lhs.size(); ++i) {
      REQUIRE (compare(lhs[i], src[i]));
    }
  }
}