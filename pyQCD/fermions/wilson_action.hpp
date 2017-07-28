#ifndef PYQCD_FERMION_WILSON_ACTION_HPP
#define PYQCD_FERMION_WILSON_ACTION_HPP
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
 * Implementation of the Wilson fermion action.
 */

#include <utils/matrices.hpp>

#include "fermion_action.hpp"
#include "hopping_matrix.hpp"


namespace pyQCD
{
  namespace fermions
  {
    template <typename Real, int Nc>
    class WilsonAction : public Action<Real, Nc>
    {
    public:
      WilsonAction(const Real mass,
                   const LatticeColourMatrix<Real, Nc>& gauge_field,
                   const std::vector<Real>& boundary_phases);

      LatticeColourVector<Real, Nc> apply_full(
          const LatticeColourVector<Real, Nc>& fermion_in) const override;
      LatticeColourVector<Real, Nc> apply_even_even_inv(
          const LatticeColourVector<Real, Nc>& fermion_in) const override;
      LatticeColourVector<Real, Nc> apply_odd_odd(
          const LatticeColourVector<Real, Nc>& fermion_in) const override;
      LatticeColourVector<Real, Nc> apply_even_odd(
          const LatticeColourVector<Real, Nc>& fermion_in) const override;
      LatticeColourVector<Real, Nc> apply_odd_even(
          const LatticeColourVector<Real, Nc>& fermion_in) const override;

      LatticeColourVector<Real, Nc> apply_hermiticity(
          const LatticeColourVector<Real, Nc>& fermion) const override;
      LatticeColourVector<Real, Nc> remove_hermiticity(
          const LatticeColourVector<Real, Nc>& fermion) const override;

    private:
      std::vector<SpinMatrix<Real>> generate_spin_structures(
          const unsigned int num_dims) const;

      LatticeColourVector<Real, Nc> multiply_chiral_gamma(
          const LatticeColourVector<Real, Nc>& fermion) const;

      HoppingMatrix<Real, Nc, 1> hopping_matrix_;
      SpinMatrix<Real> chiral_gamma_;
    };


    template <typename Real, int Nc>
    WilsonAction<Real, Nc>::WilsonAction(
        const Real mass, const LatticeColourMatrix<Real, Nc>& gauge_field,
        const std::vector<Real>& boundary_phases)
      : Action<Real, Nc>(mass, boundary_phases),
        hopping_matrix_(
            gauge_field, this->phases_,
            std::move(generate_spin_structures(gauge_field.num_dims())))
    {
      long num_spins = hopping_matrix_.num_spins();

      chiral_gamma_ = SpinMatrix<Real>::Identity(num_spins, num_spins);
      chiral_gamma_.bottomRightCorner(num_spins / 2, num_spins / 2)
          = -SpinMatrix<Real>::Identity(num_spins / 2, num_spins / 2);
    }

    template <typename Real, int Nc>
    LatticeColourVector<Real, Nc> WilsonAction<Real, Nc>::apply_full(
        const LatticeColourVector<Real, Nc>& fermion_in) const
    {
      auto fermion_out = hopping_matrix_.apply_full(fermion_in);
      fermion_out += fermion_in * (4.0 + this->mass_);
      return fermion_out;
    }


    template <typename Real, int Nc>
    LatticeColourVector<Real, Nc> WilsonAction<Real, Nc>::apply_even_even_inv(
        const LatticeColourVector<Real, Nc>& fermion_in) const
    {
      LatticeColourVector<Real, Nc> fermion_out(
          fermion_in.layout(), ColourVector<Real, Nc>::Zero(),
          fermion_in.site_size());

      auto half_vol = fermion_in.volume() / 2;
      fermion_out.segment(0, half_vol) =
          fermion_in.segment(0, half_vol) / (4.0 + this->mass_);

      return fermion_out;
    }


    template <typename Real, int Nc>
    LatticeColourVector<Real, Nc> WilsonAction<Real, Nc>::apply_odd_odd(
        const LatticeColourVector<Real, Nc>& fermion_in) const
    {
      LatticeColourVector<Real, Nc> fermion_out(
          fermion_in.layout(), ColourVector<Real, Nc>::Zero(),
          fermion_in.site_size());

      auto half_vol = fermion_in.volume() / 2;
      fermion_out.segment(half_vol, half_vol) =
          (4.0 + this->mass_) * fermion_in.segment(half_vol, half_vol);

      return fermion_out;
    }


    template <typename Real, int Nc>
    LatticeColourVector<Real, Nc> WilsonAction<Real, Nc>::apply_even_odd(
        const LatticeColourVector<Real, Nc>& fermion_in) const
    {
      return hopping_matrix_.apply_even_odd(fermion_in);
    }


    template <typename Real, int Nc>
    LatticeColourVector<Real, Nc> WilsonAction<Real, Nc>::apply_odd_even(
        const LatticeColourVector<Real, Nc>& fermion_in) const
    {
      return hopping_matrix_.apply_odd_even(fermion_in);
    }


    template <typename Real, int Nc>
    LatticeColourVector<Real, Nc> WilsonAction<Real, Nc>::multiply_chiral_gamma(
        const LatticeColourVector<Real, Nc>& fermion) const
    {
      Int volume = fermion.volume();
      Int nspins = hopping_matrix_.num_spins();

      LatticeColourVector<Real, Nc> ret(
          fermion.layout(), ColourVector<Real, Nc>::Zero(), fermion.site_size());

      for (Int site_index = 0; site_index < volume; ++site_index) {

        for (Int alpha = 0; alpha < nspins; ++alpha) {
          for (Int beta = 0; beta < nspins; ++beta) {
            ret[nspins * site_index + alpha] += chiral_gamma_.coeff(alpha, beta)
                           * fermion[nspins * site_index + beta];
          }
        }
      }

      return ret;
    }

    template <typename Real, int Nc>
    LatticeColourVector<Real, Nc> WilsonAction<Real, Nc>::apply_hermiticity(
        const LatticeColourVector<Real, Nc>& fermion) const
    {
      if (fermion.num_dims() % 2 == 1) {
        // TODO: Implement handling of odd number of dimensions
        return fermion;
      }

      return multiply_chiral_gamma(fermion);
    }


    template <typename Real, int Nc>
    LatticeColourVector<Real, Nc> WilsonAction<Real, Nc>::remove_hermiticity(
        const LatticeColourVector<Real, Nc>& fermion) const
    {
      if (fermion.num_dims() % 2 == 1) {
        return fermion;
      }

      return multiply_chiral_gamma(fermion);
    }


    template <typename Real, int Nc>
    std::vector<SpinMatrix<Real>>
    WilsonAction<Real, Nc>::generate_spin_structures(
        const unsigned int num_dims) const
    {
      const auto gammas = generate_gamma_matrices<Real>(num_dims);
      auto num_spins = static_cast<long>(std::pow(2, num_dims / 2));

      std::vector<SpinMatrix<Real>> spin_structures(
          2 * num_dims, SpinMatrix<Real>::Identity(num_spins, num_spins));

      for (unsigned long i = 0; i < gammas.size(); ++i) {
        spin_structures[2 * i] -= gammas[i];
        spin_structures[2 * i] *= -0.5;
        spin_structures[2 * i + 1] += gammas[i];
        spin_structures[2 * i + 1] *= -0.5;
      }

      return spin_structures;
    }
  }
}

#endif //PYQCD_FERMION_WILSON_ACTION_HPP
