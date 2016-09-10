#ifndef PYQCD_UPDATE_HPP
#define PYQCD_UPDATE_HPP

/*
 * This file is part of pyQCD.
 *
 * pyQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
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
 * [DOCUMENTATION]
 */

#include <core/types.hpp>
#include <gauge/gauge_action.hpp>


namespace pyQCD {

  template <typename Real, int Nc>
  class Updater
  {
  public:
    typedef LatticeColourMatrix<Real, Nc> GaugeField;

    virtual void update(GaugeField& gauge_field,
                        const unsigned int num_iter) const = 0;
  };

}

#endif // PYQCD_UPDATE_HPP