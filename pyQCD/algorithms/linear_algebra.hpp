#ifndef PYQCD_LINEAR_ALGEBRA_HPP
#define PYQCD_LINEAR_ALGEBRA_HPP
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
 * Created by Matt Spraggs on 12/02/17.
 *
 * Helper functions for LatticeColourVector linear algebra.
 */

#include <core/qcd_types.hpp>


namespace pyQCD
{
  template<typename T, typename U>
  std::complex<Real> dot_fermions(const T& psi, const U& eta)
  {
    std::complex <Real> ret(0.0, 0.0);
    for (Int i = 0; i < psi.size(); ++i) {
      ret += psi[i].dot(eta[i]);
    }
    return ret;
  }
}

#endif //PYQCD_LINEAR_ALGEBRA_HPP
