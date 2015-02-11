/* Here we just implement class functions in layout.hpp */

#include "layout.hpp"


namespace pyQCD
{
  unsigned int Layout::get_array_index(
    const std::vector<unsigned int>& site) const
  {
    // Compute the lexicographic index of the specified site and use it to
    // to get the array index (coordinate at site[0] varies slowest, that at
    // site[ndim - 1] varies fastest
    assert(site.size() == num_dims_);
    int site_index = 0;
    for (unsigned int i = 0; i < num_dims_; ++i) {
      site_index *= lattice_shape_[i];
      site_index += site[i];
    }
    return array_indices_[site_index];
  }
}