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
  class LatticeSegmentView : public LatticeObj
  {
  public:
    LatticeSegmentView(T* ptr, const unsigned int size)
        : ptr_(ptr), size_(size)
    {}

    template <typename Op, typename... Vals>
    LatticeSegmentView<T>& operator=(
        const detail::LatticeExpr<Op, Vals...>& expr)
    {
      for (unsigned int i = 0; i < size_; ++i) {
        ptr_[i] = detail::eval(i, expr);
      }

      return *this;
    }


#define LATTICE_VIEW_OPERATOR_ASSIGN_DECL(op)\
    template <typename U>\
    LatticeSegmentView<T>& operator op ## =(const U& rhs);

    LATTICE_VIEW_OPERATOR_ASSIGN_DECL(+);
    LATTICE_VIEW_OPERATOR_ASSIGN_DECL(-);
    LATTICE_VIEW_OPERATOR_ASSIGN_DECL(*);
    LATTICE_VIEW_OPERATOR_ASSIGN_DECL(/);

    T& operator[](const int i) { return ptr_[i]; }
    const T& operator[](const int i) const { return ptr_[i]; }

    unsigned int size() const { return size_; }

  private:
    T* ptr_;
    unsigned int size_;
  };



#define LATTICE_VIEW_OPERATOR_ASSIGN_IMPL(op)\
  template <typename T>\
  template <typename U>\
  LatticeSegmentView<T>& LatticeSegmentView<T>::operator op ## =(const U& rhs)\
  {\
    for (unsigned int i = 0; i < size_; ++i) {\
      ptr_[i] op ## = detail::op_assign_get_rhs(i, rhs);\
    }\
    return *this;\
  }

  LATTICE_VIEW_OPERATOR_ASSIGN_IMPL(+)
  LATTICE_VIEW_OPERATOR_ASSIGN_IMPL(-)
  LATTICE_VIEW_OPERATOR_ASSIGN_IMPL(*)
  LATTICE_VIEW_OPERATOR_ASSIGN_IMPL(/)
}

#endif //PYQCD_LATTICE_VIEW_HPP
