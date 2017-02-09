#ifndef PYQCD_FERMION_TYPES_HPP
#define PYQCD_FERMION_TYPES_HPP
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
 * Created by Matt Spraggs on 09/02/17.
 *
 * Convenience types for QCD.
 */

#include <globals.hpp>
#include "wilson_action.hpp"


namespace pyQCD
{
  namespace python
  {
    typedef fermions::Action<Real, num_colours> FermionAction;
    typedef fermions::WilsonAction<Real, num_colours> WilsonFermionAction;
  }
}

#endif //PYQCD_FERMION_TYPES_HPP
