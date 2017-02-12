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
                   const LatticeColourMatrix<Real, Nc>& gauge_field);

      void apply_full(LatticeColourVector<Real, Nc>& fermion_out,
                      const LatticeColourVector<Real, Nc>& fermion_in) const;

      void apply_hermiticity(LatticeColourVector<Real, Nc>& fermion) const;
      void remove_hermiticity(LatticeColourVector<Real, Nc>& fermion) const;

    private:
      void multiply_chiral_gamma(LatticeColourVector<Real, Nc>& fermion) const;

      HoppingMatrix<Real, Nc, 1> hopping_matrix_;
      Eigen::MatrixXcd chiral_gamma_;
    };


    template <typename Real, int Nc>
    WilsonAction<Real, Nc>::WilsonAction(
        const Real mass, const LatticeColourMatrix<Real, Nc>& gauge_field)
      : Action<Real, Nc>(mass),
        hopping_matrix_(gauge_field)
    {
      typedef Eigen::MatrixXcd SpinMat;

      auto gammas = generate_gamma_matrices(gauge_field.num_dims());
      long num_spins = hopping_matrix_.num_spins();

      std::vector<SpinMat> spin_structures(
          2 * gauge_field.num_dims(), SpinMat::Identity(num_spins, num_spins));

      for (unsigned long i = 0; i < gammas.size(); ++i) {
        spin_structures[2 * i] -= gammas[i];
        spin_structures[2 * i] *= -0.5;
        spin_structures[2 * i + 1] += gammas[i];
        spin_structures[2 * i + 1] *= -0.5;
      }

      chiral_gamma_ = Eigen::MatrixXcd::Identity(num_spins, num_spins);

      for (unsigned long i = 1; i < gammas.size(); ++i) {
        chiral_gamma_ *= gammas[i];
      }
      chiral_gamma_ *= gammas[0];

      hopping_matrix_.set_spin_structures(std::move(spin_structures));
    }

    template <typename Real, int Nc>
    void WilsonAction<Real, Nc>::apply_full(
        LatticeColourVector<Real, Nc>& fermion_out,
        const LatticeColourVector<Real, Nc>& fermion_in) const
    {
      fermion_out = fermion_in * (4.0 + this->mass_);

      hopping_matrix_.apply_full(fermion_out, fermion_in);
    }


    template <typename Real, int Nc>
    void WilsonAction<Real, Nc>::multiply_chiral_gamma(
        LatticeColourVector<Real, Nc>& fermion) const
    {
      Int volume = fermion.volume();
      Int nspins = hopping_matrix_.num_spins();

      for (Int site_index = 0; site_index < volume; ++site_index) {

        aligned_vector<ColourVector<Real, Nc>> sums(
            nspins, ColourVector<Real, Nc>::Zero());

        for (Int alpha = 0; alpha < nspins; ++alpha) {
          for (Int beta = 0; beta < nspins; ++beta) {
            sums[alpha] += chiral_gamma_.coeff(alpha, beta)
                           * fermion[nspins * site_index + beta];
          }
        }
        for (Int alpha = 0; alpha < nspins; ++alpha) {
          fermion[nspins * site_index + alpha] = sums[alpha];
        }
      }
    }

    template <typename Real, int Nc>
    void WilsonAction<Real, Nc>::apply_hermiticity(
        LatticeColourVector<Real, Nc>& fermion) const
    {
      if (fermion.num_dims() % 2 == 1) {
        // TODO: Implement handling of odd number of dimensions
        return;
      }

      multiply_chiral_gamma(fermion);
    }


    template <typename Real, int Nc>
    void WilsonAction<Real, Nc>::remove_hermiticity(
        LatticeColourVector<Real, Nc>& fermion) const
    {
      if (fermion.num_dims() % 2 == 1) {
        return;
      }

      multiply_chiral_gamma(fermion);
    }
  }
}

#endif //PYQCD_FERMION_WILSON_ACTION_HPP
