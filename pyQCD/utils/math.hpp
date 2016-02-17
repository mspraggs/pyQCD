#ifndef PYQCD_MATH_HPP
#define PYQCD_MATH_HPP
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
 *
 * This file provides basic mathematics functions and constants that don't exist
 * within the C++ standard library.
 */

#include <complex>

#include "macros.hpp"


namespace pyQCD
{
  const double pi = 3.14159265358979323846264338327950288419716939937510;
  const std::complex<double> I(0.0, 1.0);

  template <typename T, typename U>
  T mod(const T i, const U n)
  {
    // Computes positive remainder of i divided by n.
    return (i % n + n) % n;
  }

  template <typename T>
  T choose(const T n, const T r)
  {
    // Compute nCr using the factorial function
    pyQCDassert((r >= 0 and r <= n),
                std::invalid_argument("Invalid arguments to choose function."));
    T k = r;
    if (k * 2 > n) {
      k = n - k;
    }
    if (k == 0 or k == n) {
      return static_cast<T>(1);
    }
    T ret = n;
    for (T i = static_cast<T>(2); i < k + 1; ++i) {
      ret *= (n - i + 1);
      ret /= i;
    }
    return ret;
  }
}

#endif
