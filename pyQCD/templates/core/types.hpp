#ifndef PYQCD_CORE_TYPES_HPP
#define PYQCD_CORE_TYPES_HPP
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
 * Convenience types for QCD.
 */

#include <Eigen/Dense>

#include <globals.hpp>
#include "lattice.hpp"

namespace pyQCD {

  template<typename T, int N>
  using ColourMatrix = Eigen::Matrix<std::complex<T>, N, N>;
  template <typename T, int N>
  using ColourVector = Eigen::Matrix<std::complex<T>, N, 1>;
  template <typename T, int N>
  using LatticeColourMatrix = Lattice<ColourMatrix<T, N>>;
  template <typename T, int N>
  using LatticeColourVector = Lattice<ColourVector<T, N>>;

  template <typename T>
  using SU2Matrix = ColourMatrix<T, 2>;
}

namespace pyQCD {
  namespace python {
    typedef Lattice<Real> LatticeReal;
    typedef Lattice<Complex> LatticeComplex;
    {% for typedef in typedefs %}
    typedef {{ typedef.cpptype }} {{ typedef.cname }};
    {% endfor %}
  }
}
#endif
