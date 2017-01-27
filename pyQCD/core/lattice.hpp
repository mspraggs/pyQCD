#ifndef PYQCD_LATTICE_HPP
#define PYQCD_LATTICE_HPP

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
 * This file declares and defines the Lattice class, which is the fundamental
 * class for representing variables on the lattice.
 */

#include <cassert>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "lattice_expr.hpp"
#include "layout.hpp"


namespace pyQCD
{
  template <typename T>
  using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;


  template <typename T>
  class Lattice : LatticeObj
  {
  public:
    // Constructors

    Lattice(const Layout& layout, const Int site_size = 1)
      : site_size_(site_size), layout_(&layout)
    {
      data_.resize(site_size_ * layout.volume());
    }

    Lattice(const Layout& layout, const T& val, const Int site_size = 1)
      : Lattice(layout, site_size)
    {
      data_.assign(data_.size(), val);
    }
    
    Lattice(const Lattice<T>& lattice) = default;
    Lattice(Lattice<T>&& lattice) = default;

    // Element accessors

    T& operator[](const int i) { return data_[i]; }
    const T& operator[](const int i) const { return data_[i]; }

    T& operator()(const Int site, const Int elem = 0)
    { return data_[site_size_ * layout_->get_array_index(site) + elem]; }
    const T& operator()(const Int site, const Int elem = 0) const
    { return data_[site_size_ * layout_->get_array_index(site) + elem]; }
    template <typename U>
    T& operator()(const U& site, const Int elem = 0)
    { return data_[site_size_ * layout_->get_array_index(site) + elem]; }
    template <typename U>
    const T& operator()(const U& site, const Int elem = 0) const
    { return data_[site_size_ * layout_->get_array_index(site) + elem]; }

    Lattice<T>& operator=(const Lattice<T>& lattice) = default;
    Lattice<T>& operator=(Lattice<T>&& lattice) = default;

    template <typename Op, typename... Vals>
    Lattice<T>& operator=(const detail::LatticeExpr<Op, Vals...>& expr)
    {
      for (unsigned int i = 0; i < data_.size(); ++i) {
        data_[i] = detail::eval(i, expr);
      }
      return *this;
    }

    void fill(const T& rhs) { data_.assign(data_.size(), rhs); }

#define LATTICE_OPERATOR_ASSIGN_DECL(op)\
    template <typename U>\
    Lattice<T>& operator op ## =(const U& rhs);

    LATTICE_OPERATOR_ASSIGN_DECL(+);
    LATTICE_OPERATOR_ASSIGN_DECL(-);
    LATTICE_OPERATOR_ASSIGN_DECL(*);
    LATTICE_OPERATOR_ASSIGN_DECL(/);

    unsigned long size() const { return data_.size(); }
    unsigned int volume() const { return layout_->volume(); }
    unsigned int num_dims() const { return layout_->num_dims(); }
    const Site& shape() const
    { return layout_->shape(); }
    const Layout& layout() const { return *layout_; }
    Int site_size() const { return site_size_; }

  protected:
    Int site_size_;
    const Layout* layout_;
    aligned_vector<T> data_;
  };


  namespace detail
  {
    template <typename T>
    auto op_assign_get_rhs(const int i, const T& value)
      -> decltype(eval(i, value))
    {
      return eval(i, value);
    }

    template <typename T>
    auto op_assign_get_rhs(const int i, const T& value)
      -> typename std::enable_if<
        not std::is_base_of<LatticeObj, T>::value, const T&>::type
    {
      return value;
    }
  }


#define LATTICE_OPERATOR_ASSIGN_IMPL(op)\
  template <typename T>\
  template <typename U>\
  Lattice<T>& Lattice<T>::operator op ## =(const U& rhs)\
  {\
    for (unsigned int i = 0; i < data_.size(); ++i) {\
      data_[i] op ## = detail::op_assign_get_rhs(i, rhs);\
    }\
    return *this;\
  }

LATTICE_OPERATOR_ASSIGN_IMPL(+);
LATTICE_OPERATOR_ASSIGN_IMPL(-);
LATTICE_OPERATOR_ASSIGN_IMPL(*);
LATTICE_OPERATOR_ASSIGN_IMPL(/);
}

#endif
