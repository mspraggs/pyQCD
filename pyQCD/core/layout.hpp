#ifndef PYQCD_LAYOUT_HPP
#define PYQCD_LAYOUT_HPP

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
 * This file provides a base Layout classes and derived classes that specify
 * the layout of lattice sites in memory. These classes are then used in Lattice
 * objects to refer to the correct lattice site within the data_ member
 * variable.
 *
 * The majority of the code in this file belongs to the Layout class, which
 * accepts a function in its constructor. This function takes a lexicographic
 * site index and returns the array index corresponding to that site. This way
 * a pair of std::vectors can be created that take one from memory space to
 * lattice space, and vice versa.
 *
 * All sub-classes of the Layout class must pass a function to the delegated
 * Layout constructor.
 */

#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

#include <utils/math.hpp>


namespace pyQCD
{
  typedef unsigned int Int;
  typedef std::vector<Int> Site;

  class Layout
  {
  public:

    Layout() = default;
    Layout(const Site& shape)
      : num_dims_(static_cast<Int>(shape.size())), shape_(shape)
    {
      // Constructor create arrays of site/array indices
      volume_ = std::accumulate(shape.begin(), shape.end(), 1u,
        std::multiplies<Int>());
    }
    virtual ~Layout() = default;

    // Functions to retrieve array indices and so on.
    template <typename T,
      typename std::enable_if<not std::is_integral<T>::value>::type* = nullptr>
    inline Int get_array_index(const T& site) const;
    inline Int get_array_index(const Int site_index) const
    { return array_indices_[site_index]; }
    inline Int get_site_index(const Int array_index) const
    { return site_indices_[array_index]; }
    inline Site compute_site_coords(const Int site_index) const;
    template <typename T>
    inline void sanitize_site_coords(T& coords) const;

    Int volume() const { return volume_; }
    Int num_dims() const { return num_dims_; }
    const std::vector<Int>& shape() const
    { return shape_; }

  protected:
    Int num_dims_, volume_;
    Site shape_;
    // array_indices_[site_index] -> array_index
    std::vector<Int> array_indices_;
    // site_indices_[array_index] -> site_index
    std::vector<Int> site_indices_;
  };


  class LexicoLayout : public Layout
  {
  public:
    LexicoLayout(const Site& shape) : Layout(shape)
    {
      array_indices_.resize(volume_);
      site_indices_.resize(volume_);
      for (Int i = 0; i < volume_; ++i) {
        array_indices_[i] = i;
        site_indices_[i] = i;
      }
    }
  };


  template <typename T,
    typename std::enable_if<not std::is_integral<T>::value>::type*>
  inline Int Layout::get_array_index(const T& site) const
  {
    // Compute the lexicographic index of the specified site and use it to
    // to get the array index (coordinate at site[0] varies slowest, that at
    // site[ndim - 1] varies fastest
    int site_index = 0;
    for (Int i = 0; i < num_dims_; ++i) {
      site_index *= shape_[i];
      site_index += site[i];
    }
    return array_indices_[site_index];
  }

  inline Site Layout::compute_site_coords(const Int site_index) const
  {
    // Compute the coordinates of the site specified by the given index
    auto site_index_copy = site_index;
    Site ret(num_dims_);
    for (int i = num_dims_ - 1; i > -1; --i) {
      ret[i] = mod(site_index_copy, shape_[i]);
      site_index_copy /= shape_[i];
    }
    return ret;
  }

  template <typename T>
  inline void Layout::sanitize_site_coords(T& coords) const
  {
    // Apply periodic boundary conditions to coords
    for (unsigned i = 0; i < num_dims_; ++i) {
      coords[i] = mod(coords[i], shape_[i]);
    }
  }
}

#endif