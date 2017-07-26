#ifndef PYQCD_GAUGE_ACTION_HPP
#define PYQCD_GAUGE_ACTION_HPP

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
 * This file contains the base implementation of the base gauge action type,
 * upon which all gauge actions are based.
 */

#include <core/qcd_types.hpp>


namespace pyQCD
{
  namespace gauge
  {
    template <typename Real, int Nc>
    class Action
    {
    public:
      using GaugeLink = ColourMatrix<Real, Nc>;
      using GaugeField = LatticeColourMatrix<Real, Nc>;

      Action(const Real beta) : beta_(beta) { }

      virtual ~Action() = default;

      virtual GaugeLink compute_staples(const GaugeField& gauge_field,
                                        const Int site_index) const = 0;

      virtual Real local_action(const GaugeField& gauge_field,
                                const Int site_index) const = 0;

      inline Real beta() const { return beta_; }

    private:
      Real beta_; // The inverse coupling
    };
  }
}

#endif // PYQCD_GAUGE_ACTION_HPP
