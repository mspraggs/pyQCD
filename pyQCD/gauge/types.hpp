#ifndef PYQCD_GAUGE_TYPES_HPP
#define PYQCD_GAUGE_TYPES_HPP
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
 * Created by Matt Spraggs on 14/01/17.
 *
 *
 * Convenience types for QCD.
 */

#include <globals.hpp>
#include <gauge/gauge_action.hpp>
#include <gauge/rectangle_action.hpp>
#include <gauge/wilson_action.hpp>


namespace pyQCD {
  namespace python {
    typedef gauge::Action<Real, num_colours> GaugeAction;
    typedef gauge::WilsonAction<Real, num_colours> WilsonGaugeAction;
    typedef gauge::RectangleAction<Real, num_colours> RectangleGaugeAction;
  }
}

#endif //PYQCD_GAUGE_TYPES_HPP