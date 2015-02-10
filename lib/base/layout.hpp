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
    Layout(const std::vector<unsigned int>& shape);
    virtual ~Layout() = default;

    unsigned int get_array_index(const unsigned int site_index) const
    { return array_indices_[site_index]; }
    unsigned int get_array_index(const std::vector<unsigned int>& site) const;
    unsigned int get_site_index(const unsigned int array_index) const
    { return site_indices_[array_index]; }

  protected:
    virtual unsigned int compute_array_index(
      const unsigned int site_index) const = 0;

  private:
    unsigned int num_dims_, lattice_volume_;
    std::vector<unsigned int> lattice_shape_;
    // indices_[site_index] -> array_index
    std::vector<unsigned int> array_indices_;
    // site_indices_[array_index] -> site_index
    std::vector<unsigned int> site_indices_;
  };
}

#endif