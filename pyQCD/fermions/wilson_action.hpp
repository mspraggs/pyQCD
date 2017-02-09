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

    private:
      HoppingMatrix<Real, Nc, 1> hopping_matrix_;
    };


    template <typename Real, int Nc>
    WilsonAction<Real, Nc>::WilsonAction(
        const Real mass, const LatticeColourMatrix<Real, Nc>& gauge_field)
      : Action<Real, Nc>(mass),
        hopping_matrix_(gauge_field)
    {
      typedef Eigen::MatrixXcd SpinMat;

      auto gammas = generate_gamma_matrices(gauge_field.num_dims());
      long mat_size = gammas[0].cols();
      std::vector<SpinMat> spin_structures(
          2 * gauge_field.num_dims(), SpinMat::Identity(mat_size, mat_size));

      for (unsigned long i = 0; i < gammas.size(); ++i) {
        spin_structures[2 * i] -= gammas[i];
        spin_structures[2 * i] *= -0.5;
        spin_structures[2 * i + 1] += gammas[i];
        spin_structures[2 * i + 1] *= -0.5;
      }

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
  }
}

#endif //PYQCD_FERMION_WILSON_ACTION_HPP
