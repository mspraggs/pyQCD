#ifndef PYQCD_MACROS_HPP
#define PYQCD_MACROS_HPP
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
 * This file provides macros to help neaten up code and encapsulate various
 * preprocessor bits and pieces.
 */

#include <iostream>

// Custom assert command - Cython can process this.
#ifndef NDEBUG
#define pyQCDassert(expr, exception)                            \
if (not (expr)) {                                               \
  std::cout << "Assertion " << #expr << " failed" << std::endl; \
  throw exception;                                              \
}
#else
#define pyQCDassert(expr, exception)
#endif

#endif