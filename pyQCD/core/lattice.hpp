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

#include "aligned_allocator.hpp"
#include "lattice_expr.hpp"
#include "lattice_view.hpp"
#include "layout.hpp"


namespace pyQCD
{
  template <typename T>
  using aligned_vector = std::vector<T, detail::aligned_allocator<T>>;


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

    // Assignment operators

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

#define LATTICE_OPERATOR_ASSIGN_DECL(op)\
    template <typename U>\
    Lattice<T>& operator op ## =(const U& rhs);

    LATTICE_OPERATOR_ASSIGN_DECL(+);
    LATTICE_OPERATOR_ASSIGN_DECL(-);
    LATTICE_OPERATOR_ASSIGN_DECL(*);
    LATTICE_OPERATOR_ASSIGN_DECL(/);

    void fill(const T& rhs) { data_.assign(data_.size(), rhs); }

    void change_layout(const Layout& new_layout);

    LatticeSegmentView<T> segment(const unsigned int offset,
                                  const unsigned int size);
    LatticeSegmentView<const T> segment(const unsigned int offset,
                                        const unsigned int size) const;

    unsigned long size() const { return data_.size(); }
    unsigned int volume() const { return layout_->volume(); }
    unsigned int num_dims() const { return layout_->num_dims(); }
    const Site& shape() const { return layout_->shape(); }
    const Layout& layout() const { return *layout_; }
    Int site_size() const { return site_size_; }

  protected:
    Int site_size_;
    const Layout* layout_;
    aligned_vector<T> data_;

  private:
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

LATTICE_OPERATOR_ASSIGN_IMPL(+)
LATTICE_OPERATOR_ASSIGN_IMPL(-)
LATTICE_OPERATOR_ASSIGN_IMPL(*)
LATTICE_OPERATOR_ASSIGN_IMPL(/)

  template <typename T>
  void Lattice<T>::change_layout(const Layout& new_layout)
  {
    // Assigns new Layout instance and reorders existing data in place.

    if (&new_layout == layout_) {
      return;
    }

    // Construct permutation array specifying the reordering
    std::vector<Int> array_permutations(layout_->volume());

    for (Int i = 0; i < layout_->volume(); ++i) {
      auto index_old = layout_->get_array_index(i);
      auto index_new = new_layout.get_array_index(i);
      array_permutations[index_new] = index_old;
    }

    // Temporary store for site data to prevent data loss
    aligned_vector<T> site_data_store(site_size_);

    // Move along the permutation array in order and move data as specified
    // by the permutation
    for (Int i = 0; i < layout_->volume(); ++i) {

      // Save the current site's data for use later
      site_data_store.assign(&data_[site_size_ * i],
                             &data_[site_size_ * (i + 1)]);

      Int j = i;

      // Now we follow the array indices around the list of permutataions,
      // sorting the permutation array in the process. Each time we loop, we
      // check for cycles in the permutations array by checking if we're back
      // at the index i.
      while (true) {
        Int k = array_permutations[j];
        array_permutations[j] = j;

        // Break if we've completed a cycle.
        if (k == i) {
          break;
        }

        std::copy(&data_[site_size_ * k], &data_[site_size_ * (k + 1)],
                  &data_[site_size_ * j]);
        j = k;
      }

      std::copy(site_data_store.begin(), site_data_store.end(),
                &data_[site_size_ * j]);
    }

    layout_ = &new_layout;
  }

  template <typename T>
  LatticeSegmentView<T> Lattice<T>::segment(const unsigned int offset,
                                            const unsigned int size)
  {
    return LatticeSegmentView<T>(&data_[offset * site_size_],
                                 size * site_size_);
  }

  template <typename T>
  LatticeSegmentView<const T> Lattice<T>::segment(const unsigned int offset,
                                                  const unsigned int size) const
  {
    return LatticeSegmentView<const T>(&data_[offset * site_size_],
                                       size * site_size_);
  }
}

#endif
