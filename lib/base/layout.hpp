#ifndef LAYOUT_HPP
#define LAYOUT_HPP

/* This file provides a base Layout classes and derived classes that specify
 * the layout of lattice sites. These classes are then used in Lattice objects
 * and their derived types to specify the relationship between e
 */

#include <cassert>
#include <vector>


namespace pyQCD
{
  class Layout
  {
  public:
    Layout(std::vector<int>& shape);
    virtual ~Layout() = default;

    int get_array_index(const int site_index) const
    { return array_indices_[site_index]; }
    int get_array_index(const std::vector<int>& site) const;
    int get_site_index(const int array_index) const
    { return site_indices_[array_index]; }

  protected:
    virtual int compute_array_index(const int site_index) const = 0;

  private:
    int num_dims_, lattice_volume_;
    std::vector<int> lattice_shape_;
    // indices_[site_index] -> array_index
    std::vector<int> array_indices_;
    // site_indices_[array_index] -> site_index
    std::vector<int> site_indices_;
  };
}

#endif