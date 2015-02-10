/* Here we just implement class functions in layout.hpp */

#include <functional>
#include <numeric>

#include "layout.hpp"


namespace pyQCD
{
  Layout::Layout(const std::vector<unsigned int>& lattice_shape)
    : num_dims_(static_cast<unsigned int>(lattice_shape.size())),
      lattice_shape_(lattice_shape)
  {
    // Constructor create arrays of site/array indices
    lattice_volume_ = std::accumulate(lattice_shape.begin(),
                                      lattice_shape.end(),
                                      unsigned(1),
                                      std::multiplies<unsigned int>());

    array_indices_.resize(lattice_volume_);
    site_indices_.resize(lattice_volume_);
    for (unsigned int site_index = 0;
         site_index < lattice_volume_;
         ++site_index) {
      unsigned int array_index = compute_array_index(site_index);
      array_indices_[site_index] = array_index;
      site_indices_[array_index] = site_index;
    }
  }


  unsigned int Layout::get_array_index(
    const std::vector<unsigned int>& site) const
  {
    // Compute the lexicographic index of the specified site and use it to
    // to get the array index (coordinate at site[0] varies slowest, that at
    // site[ndim - 1] varies fastest
    assert(site.size() == num_dims_);
    int site_index = 0;
    for (int i = 0; i < num_dims_; ++i) {
      site_index *= lattice_shape_[i];
      site_index += site[i];
    }
    return array_indices_[site_index];
  }
}