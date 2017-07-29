#ifndef PYQCD_MATH_HPP
#define PYQCD_MATH_HPP
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
 * This file provides basic mathematics functions and constants that don't exist
 * within the C++ standard library.
 */

#include <complex>


namespace pyQCD
{
  constexpr double pi = 3.14159265358979323846264338327950288419716939937510;
  constexpr std::complex<double> I(0.0, 1.0);

  template <typename T, typename U>
  auto mod(const T i, const U n) -> decltype((i % n + n) % n)
  {
    return (i % n + n) % n;
  }
}

#endif
