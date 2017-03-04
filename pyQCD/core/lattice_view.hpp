#ifndef PYQCD_LATTICE_VIEW_HPP
#define PYQCD_LATTICE_VIEW_HPP

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
 * Created by Matt Spraggs on 04/03/17.
 *
 * Below we define views on the Lattice class.
 */

#include <vector>

#include "lattice_expr.hpp"


namespace pyQCD {

  template <typename T>
  class LatticeView : public LatticeObj
  {
  public:
    LatticeView(const std::vector<T*>& view_data) : view_data_(view_data) {}
    LatticeView(std::vector<T*>&& view_data) : view_data_(view_data) {}

    T& operator[](const int i) { return *view_data_[i]; }
    const T& operator[](const int i) const { return *view_data_[i]; }

  private:
    std::vector<T*> view_data_;
  };
}

#endif //PYQCD_LATTICE_VIEW_HPP
