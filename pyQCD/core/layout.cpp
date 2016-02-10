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
 */

#include <algorithm>

#include <utils/macros.hpp>

#include "layout.hpp"


namespace pyQCD
{

#ifndef USE_MPI
  Layout::Layout(const Site& shape, const Layout::ArrFunc& compute_array_index)
  {
    // Constructor create arrays of site/array indices
    local_volume_ = std::accumulate(shape.begin(), shape.end(), 1u,
                                    std::multiplies<Int>());

    array_indices_local_.resize(local_volume_);
    site_indices_.resize(local_volume_);
    for (Int site_index = 0; site_index < local_volume_; ++site_index) {
      Int array_index = compute_array_index(site_index);
      array_indices_local_[site_index] = array_index;
      site_indices_[array_index] = site_index;
    }
  }
#else
  Layout::Layout(const Site& shape, const ArrFunc& compute_array_index,
                 const Site& partition, const Int halo_depth,
                 const Int max_mpi_hop)
    : num_dims_(static_cast<Int>(shape.size())),
      global_shape_(shape), partition_(partition)
  {
    for (size_t i = 0; i < num_dims_; ++i) {
      pyQCDassert((shape[i] % partition[i] == 0),
                  std::logic_error("Supplied lattice shape not divisible by "
                                   "MPI partition."))
      local_shape_[i] = shape[i] / partition[i];
    }

    global_volume_ = std::accumulate(shape.begin(), shape.end(), 1u,
                                     std::multiplies<Int>());
    local_volume_ = std::accumulate(local_shape_.begin(), local_shape_.end(),
                                    1u, std::multiplies<Int>());
  }
#endif
  Int Layout::local_to_global_array_index(const Int index) const
  {
    // Move from local to global array index
    auto site_index = site_indices_[index];
    auto site = compute_site_coords(site_index);
  }
}