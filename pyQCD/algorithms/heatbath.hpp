#ifndef PYQCD_HEATBATH_HPP
#define PYQCD_HEATBATH_HPP

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
 * This file contains the functions necessary to update a single gauge link
 * using the pseudo heatbath algorithm.
 */

#include <array>
#include <random>

#include <core/qcd_types.hpp>
#include <gauge/gauge_action.hpp>
#include <utils/matrices.hpp>
#include <utils/random.hpp>


namespace pyQCD {

  template <typename Real>
  SU2Matrix<Real> gen_heatbath_su2(RandGenerator& rng, const Real weight)
  {
    // Generate a random SU(2) matrix distributed according to the distribution
    // exp(0.5 * weight * beta * Re tr(X)). We use the algorithm specified in
    // Kennedy and Pendleton (1985), in Phys. Lett. 156B.
    //
    // (See also page 87 of Gattringer and Lang for the source material for
    // this algorithm.)

    // Coefficients for the SU(2) basis of Pauli matrices. This final vector
    // will need to be normalised.
    std::array<Real, 4> coeffs{0.0, 0.0, 0.0, 0.0};
    // Now we need to fill the components of this vector. The first component
    // must be distributed according to
    //   sqrt(1 - x^2) * exp(weight * beta * x)
    Real lambda_squared = 2.0;
    Real uniform_squared = std::pow(rng.generate_real<Real>(0.0, 1.0), 2);
    while (uniform_squared > 1 - lambda_squared) {
      const std::array<Real, 3> r {
          1.0 - rng.generate_real<Real>(0.0, 1.0),
          1.0 - rng.generate_real<Real>(0.0, 1.0),
          1.0 - rng.generate_real<Real>(0.0, 1.0)
      };

      lambda_squared = - 1.0 / (2.0 * weight) *
          (std::log(r[0]) + std::pow(std::cos(2 * pi * r[1]), 2) *
                                std::log(r[2]));
      uniform_squared = std::pow(rng.generate_real<Real>(0.0, 1.0), 2);
    }
    coeffs[0] = 1 - 2 * lambda_squared;
    // With the first component determined, the magnitude of the remaining
    // three-vector can easily be determined.
    const Real three_vec_magnitude = std::sqrt(1 - coeffs[0] * coeffs[0]);
    // The remaining three-vector should then be take from a uniform spherical
    // distribution.
    const Real cos_theta = rng.generate_real<Real>(-1.0, 1.0);
    const Real sin_theta = std::sqrt(1 - cos_theta * cos_theta);
    const Real phi = rng.generate_real<Real>(0, 2 * pi);

    coeffs[1] = three_vec_magnitude * sin_theta * std::cos(phi);
    coeffs[2] = three_vec_magnitude * sin_theta * std::sin(phi);
    coeffs[3] = three_vec_magnitude * cos_theta;

    return construct_su2<Real>(coeffs);
  }

  template <typename Real, int Nc>
  ColourMatrix<Real, Nc> comp_su2_heatbath_mat(
      RandGenerator& rng, const ColourMatrix<Real, Nc>& W, const Real weight,
      const unsigned int subgroup)
  {
    // Perform an SU(2) heatbath update on the given lattice link
    auto A = extract_su2(W, subgroup);
    const auto sqrt_detA = std::sqrt(A.determinant());
    A /= sqrt_detA;
    const Real a = sqrt_detA.real();
    bool det_is_zero = a < 6.0 * std::numeric_limits<Real>::epsilon();
    const auto X =
        det_is_zero ? random_su2<Real>(rng) : gen_heatbath_su2(rng, a * weight);
    return insert_su2<Nc>((X * A.adjoint()).eval(), subgroup);
  }

  template <typename Real, int Nc>
  void heatbath_link_update(RandGenerator& rng,
                            LatticeColourMatrix<Real, Nc> &gauge_field,
                            const gauge::Action<Real, Nc> &action,
                            const Int link_index)
  {
    // Perform SU(N) heatbath update on the specified lattice link
    const auto staple = action.compute_staples(gauge_field, link_index);
    auto& link = gauge_field(link_index / gauge_field.site_size(),
                             link_index % gauge_field.site_size());
    const Real beta_prime = action.beta() / Nc;

    constexpr int num_subgroups = (Nc * (Nc - 1)) / 2;

    // Here we do the pseudo-heatbath over the subgroups of SU(N)
    for (unsigned int subgroup = 0; subgroup < num_subgroups; ++subgroup) {
      const ColourMatrix<Real, Nc> link_prod = link * staple;
      link = comp_su2_heatbath_mat(rng, link_prod, beta_prime, subgroup) * link;
    }
  }


  template <typename Real, int Nc>
  void heatbath_update(LatticeColourMatrix<Real, Nc>& gauge_field,
                       const gauge::Action<Real, Nc>& action,
                       const unsigned int num_iter)
  {
    auto num_links = gauge_field.size();
    auto site_size = gauge_field.site_size();
    auto& random_wrapper = rng(gauge_field.layout());
    for (unsigned int i = 0; i < num_iter; ++i) {
      for (unsigned int link = 0; link < num_links; ++link) {
        auto& rng = random_wrapper[link / site_size];
        heatbath_link_update(rng, gauge_field, action, link);
      }
    }
  }


  template <typename Real, int Nc>
  class Heatbath
  {
  public:
    Heatbath(const Layout& layout, const gauge::Action<Real, Nc>& action);

    void update(LatticeColourMatrix<Real, Nc>& gauge_field,
                const unsigned int num_iter);

  private:
    RandomWrapper* rngs_;
    const gauge::Action<Real, Nc>* action_;
    std::vector<std::vector<Int>> site_partitioning_;
  };


  template <typename Real, int Nc>
  Heatbath<Real, Nc>::Heatbath(const Layout& layout,
                               const gauge::Action<Real, Nc>& action)
      : rngs_(&RandomWrapper::instance(layout)), action_(&action)
  {
    // Here we partition the sites such that the sites in a given partition
    // form a rhombic lattice with the shortest lattice edge given by
    // action.min_site_diag_offset(). This is achieved in two stages.
    //
    // First, the separation parameter is used in conjunction with modular
    // arithmetic to generate an educated guess for the partitions. This
    // educated guess is generated by ordering the sites according to the
    // expected partition of each site.
    //
    // Second, this site ordering is given to a greedy multicolouring algorithm,
    // which accounts for complications associated with generating the guess,
    // such as the separation parameter not being a factor of one of the lattice
    // extents.

    // Stage one: guess partition site ordering
    std::vector<unsigned int> site_ordering(layout.volume());
    std::iota(site_ordering.begin(), site_ordering.end(), 0);

    PartitionCompare compare(action.min_site_diag_offset(), layout);
    std::sort(site_ordering.begin(), site_ordering.end(), compare);

    // Stage two: generate partitioning using guessed ordering
    site_partitioning_ = layout.partition_sites(
        [&action] (const Int idx, const Layout& loc_layout) {
          return action.participating_sites(idx, loc_layout);
        }, site_ordering);
  }


  template <typename Real, int Nc>
  void Heatbath<Real, Nc>::update(LatticeColourMatrix<Real, Nc>& gauge_field,
                                  const unsigned int num_iter)
  {
    const auto num_dims = gauge_field.site_size();

    for (unsigned int it = 0; it < num_iter; ++it) {
      for (const auto& partition : site_partitioning_) {
        for (unsigned int mu = 0; mu < num_dims; ++mu) {
#pragma omp parallel for
          for (unsigned int idx = 0; idx < partition.size(); ++idx) {
            const auto site = partition[idx];
            auto& rng = (*rngs_)[site];
              heatbath_link_update(rng, gauge_field, *action_,
                                   num_dims * site + mu);
          }
        }
      }
    }
  }
}

#endif
