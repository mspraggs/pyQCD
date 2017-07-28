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
 * Tests for the heatbath update algorithm.
 */

#include <algorithms/heatbath.hpp>

#include "helpers.hpp"


template <typename Real, int Nc>
class TestGaugeAction : public pyQCD::gauge::Action<Real, Nc>
{
public:
  using GaugeLink =  typename pyQCD::gauge::Action<Real, Nc>::GaugeLink;
  using GaugeField = typename pyQCD::gauge::Action<Real, Nc>::GaugeField;

  TestGaugeAction(const Real beta)
    : pyQCD::gauge::Action<Real, Nc>(beta)
  { }

  GaugeLink compute_staples(const GaugeField& gauge_field,
                            const pyQCD::Int site_index) const override
  {
    return GaugeLink::Identity();
  };

  Real local_action(const GaugeField& gauge_field,
                    const pyQCD::Int site_index) const override
  { return 0.0; }
};


using Real = double;


TEST_CASE("Heatbath test")
{
  SECTION ("Test heatbath SU(2) generation") {

    Compare<Real> comp(1.0e-5, 1.0e-8);
    MatrixCompare<pyQCD::SU2Matrix<Real>> mat_comp(1.0e-5, 1.0e-8);

    const unsigned int n = 10000;
    std::vector<Real> x0s(n);
    for (unsigned int i = 0; i < n; ++i) {
      auto heatbath_su2 = pyQCD::gen_heatbath_su2(5.0);

      REQUIRE(mat_comp(heatbath_su2 * heatbath_su2.adjoint(),
        pyQCD::SU2Matrix<Real>::Identity()));
      auto det = heatbath_su2.determinant();
      REQUIRE(comp(det.real(), 1.0));
      REQUIRE(comp(det.imag(), 0.0));

      x0s[i] = heatbath_su2.trace().real() / 2.0;
    }
    // Compute the mean and the standard deviation of x0 (coefficient on
    // sigma0).
    Real mean = std::accumulate(x0s.begin(), x0s.end(), 0.0) / n;

    std::vector<Real> square_devs(n);
    std::transform(x0s.begin(), x0s.end(), square_devs.begin(),
      [mean](const Real val) { return (val - mean) * (val - mean); });
    Real sum_square_devs
      = std::accumulate(square_devs.begin(), square_devs.end(), 0.0);
    Real stddev = std::sqrt(sum_square_devs / n);

    Compare<Real> comp_weak(0.005, 0.005);
    REQUIRE(comp_weak(mean, 0.7193405813643129));
    REQUIRE(comp_weak(stddev, 0.2257095017580442));
  }

  SECTION ("Testing SU(2) heatbath update") {

    constexpr int Nc = 3;

    using ColourMatrix = pyQCD::ColourMatrix<Real, Nc>;

    Compare<Real> comp(1.0e-5, 1.0e-8);
    MatrixCompare<ColourMatrix> mat_comp(1.0e-5, 1.0e-8);

    for (unsigned int subgroup = 0; subgroup < Nc; ++subgroup) {
      ColourMatrix link = ColourMatrix::Identity();
      ColourMatrix staple = ColourMatrix::Identity();

      ColourMatrix link_prod = link * staple;
      const auto update_matrix =
          pyQCD::comp_su2_heatbath_mat(link_prod, 5.0, subgroup);
      link = update_matrix * link;

      REQUIRE(comp(link(2 - subgroup, 2 - subgroup).real(), 1.0));
      REQUIRE(comp(link(2 - subgroup, 2 - subgroup).imag(), 0.0));

      REQUIRE(comp(link.determinant().real(), 1.0));
      REQUIRE(comp(link.determinant().imag(), 0.0));
      REQUIRE(mat_comp(link * link.adjoint(), ColourMatrix::Identity()));
    }
  }

  SECTION ("Testing SU(3) heatbath update") {

    using ColourMatrix = pyQCD::ColourMatrix<Real, 3>;

    auto layout = pyQCD::LexicoLayout({8, 4, 4, 4, 4});
    auto gauge_field
      = pyQCD::LatticeColourMatrix<Real, 3>(layout, ColourMatrix::Identity());

    auto action = TestGaugeAction<Real, 3>(5.0);

    pyQCD::heatbath_link_update(gauge_field, action, 0);

    auto link = gauge_field(0);

    Compare<Real> comp(1.0e-5, 1.0e-8);
    MatrixCompare<ColourMatrix> mat_comp(1.0e-5, 1.0e-8);
    auto det = link.determinant();
    REQUIRE(comp(det.real(), 1.0));
    REQUIRE(comp(det.imag(), 0.0));
    REQUIRE(mat_comp(link.adjoint() * link, ColourMatrix::Identity()));
  }
}