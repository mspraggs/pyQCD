#ifndef PYQCD_FERMION_ACTION_HPP
#define PYQCD_FERMION_ACTION_HPP
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
 * Base class for all gauge action types.
 */

#include <core/types.hpp>


namespace pyQCD
{
  namespace fermions
  {
    template <typename Real, int Nc>
    class Action
    {
    public:
      Action(const Real mass) : mass_(mass) {}

      virtual void apply_full(
          LatticeColourVector<Real, Nc>& fermion_out,
          const LatticeColourVector<Real, Nc>& fermion_in) const = 0;

      virtual void apply_hermiticity(
          LatticeColourVector<Real, Nc>& fermion) const = 0;
      virtual void remove_hermiticity(
          LatticeColourVector<Real, Nc>& fermion) const = 0;

    protected:
      Real mass_;
    };
  }
}

#endif //PYQCD_FERMION_ACTION_HPP
