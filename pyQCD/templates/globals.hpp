#ifndef PYQCD_GLOBALS_HPP
#define PYQCD_GLOBALS_HPP
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
 * Created by Matt Spraggs on 17/01/17.
 *
 * This file contains package-wide variables and type definitions.
 */

#include <complex>

namespace pyQCD
{
  constexpr int num_colours = {{ num_colours }};
  typedef {{ precision }} Real;
  typedef std::complex<Real> Complex;
}

#endif //PYQCD_GLOBALS_HPP
