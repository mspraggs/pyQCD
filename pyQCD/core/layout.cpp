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

#include <Eigen/Dense>

#include <utils/macros.hpp>

#include "layout.hpp"


namespace pyQCD
{

#ifndef USE_MPI
  Layout::Layout(const Site& shape, const Layout::ArrFunc& compute_array_index)
  {
    // Constructor create arrays of site/array indices
    local_volume_ = detail::compute_volume(shape.begin(), shape.size());

    site_indices_.resize(local_volume_);
    for (Int site_index = 0; site_index < local_volume_; ++site_index) {
      Int array_index = compute_array_index(site_index);
      array_indices_local_[site_index] = array_index;
      site_indices_[array_index] = site_index;
    }
  }
#else
  Layout::Layout(const Site& shape, const Site& partition, const Int halo_depth,
                   const Int max_mpi_hop)
    : num_dims_(static_cast<Int>(shape.size())),
      global_shape_(shape), partition_(partition), halo_depth_(halo_depth)
  {
    // Initialize the MPI layout and associated halo buffers

    //---------------------- Compute some basic layout info --------------------

    local_shape_.resize(num_dims_);

    for (size_t i = 0; i < num_dims_; ++i) {
      pyQCDassert((shape[i] % partition[i] == 0),
                  std::logic_error("Supplied lattice shape not divisible by "
                                   "MPI partition."))
      local_shape_[i] = shape[i] / partition[i];
    }
    auto num_comm_dims = std::count_if(partition.begin(), partition.end(),
                                       [] (const Int p) { return p > 1; });
    pyQCDassert((num_comm_dims >= max_mpi_hop),
                std::logic_error("Supplied max_mpi_hop must be smaller "
                                   "than all local shape extents"))

    global_volume_ = detail::compute_volume(shape.begin(), shape.end(), 1u);
    local_volume_ = detail::compute_volume(local_shape_.begin(),
                                           local_shape_.end(), 1u);

    //---------------------- Get some basic MPI info ---------------------------
    initialise_buffers(partition, max_mpi_hop);
  }

  void Layout::initialise_buffers(const Site& partition, const Int max_mpi_hop)
  {
    /* We need to do a few things here:
     * - Determine some indexing strategy for the various buffers
     * - Determine MPI ranks of neighbours corresponding to buffers
     * - Arrange buffers such that the largest are stored first
     *   - A result of this is that buffers with the fewest mpi hops will be
     *     stored first.
     * - Determine the shapes of the mpi buffers
     * - Determine the volumes of the buffers
     * - Compute the following mappings:
     *   - from local lexicographic index (w.r.t. local shape including the
     *     halo) to array index;
     *   - from array index to lexicographic index;
     *   - global lexicographic index to local array index (through some
     *     std::unordered_map or something).
     * - Create a list of array indexes specifying sites that are going to end
     *   up in buffers on neighbouring nodes. Note this is probably going to be
     *   less fun than pulling teeth.
     */

    // Determine dimensions where we need comms.
    std::vector<bool> need_comms(num_dims_, true);
    std::transform(partition.begin(), partition.end(), need_comms.begin(),
                   [] (const Int d) { return d > 1; });

    Site local_shape_with_halo = local_shape_;
    for (Int i = 0; i < num_dims_; ++i) {
      if (need_comms[i]) {
        local_shape_with_halo[i] += 2 * halo_depth_;
      }
    }
    PYQCD_TRACE
    local_size_ = detail::compute_volume(local_shape_with_halo.begin(),
                                         local_shape_with_halo.end(), 1u);
    array_indices_local_.resize(local_size_);
    site_indices_.resize(local_size_);
    // Get the MPI coordinate of this node.
    detail::IVec mpi_coord(num_dims_);
    MPI_Cart_coords(Communicator::instance().comm(),
                    Communicator::instance().rank(),
                    static_cast<int>(num_dims_), mpi_coord.data());

    detail::IVec local_shape = detail::site_to_ivec(local_shape_);

    detail::IVec local_origin
      = (local_shape.array() * mpi_coord.array()).matrix();
    PYQCD_TRACE

    auto mpi_offsets = detail::generate_mpi_offsets(max_mpi_hop, need_comms);

    auto offset_sort = [] (const detail::IVec& vec1, const detail::IVec& vec2) {
      return vec1.squaredNorm() < vec2.squaredNorm();
    };

    std::stable_sort(mpi_offsets.begin(), mpi_offsets.end(), offset_sort);
    num_buffers_ = static_cast<Int>(mpi_offsets.size());

    PYQCD_TRACE
    buffer_ranks_.resize(mpi_offsets.size());
    buffered_site_indices_.resize(mpi_offsets.size());
    buffer_volumes_.resize(mpi_offsets.size());

    // Strap yourself in, things are about to get ugly...
    // TODO: Clear this up/optimize as much as possible.
    buffer_map_.resize(2 * num_dims_);
    for (Int dim = 0; dim < num_dims_; ++dim) {
      auto axis = compute_axis(dim, MpiDirection::FRONT);
      buffer_map_[axis].resize(max_mpi_hop);
      axis = compute_axis(dim, MpiDirection::BACK);
      buffer_map_[axis].resize(max_mpi_hop);
    }

    auto& comm = Communicator::instance().comm();
    PYQCD_TRACE

    detail::IVec unbuffered_region_corner = detail::IVec::Zero(num_dims_);
    std::replace_copy_if(need_comms.begin(), need_comms.end(),
                         unbuffered_region_corner.data(),
                         [] (const bool val) { return val; }, halo_depth_);
    auto unbuffered_site_iter = detail::SiteIterator(local_shape_);
    Int array_index = 0;
    for (auto& site : unbuffered_site_iter) {
      detail::IVec local_site
        = detail::site_to_ivec(site) + unbuffered_region_corner;
      Int lex_index = detail::coords_to_lex_index(local_site,
                                                  local_shape_with_halo,
                                                  num_dims_);
      array_indices_local_[lex_index] = array_index++;
    }

    // Now loop through the various MPI offsets and compute the neighbour ranks
    Int buffer_index = 0;
    Int buffer_array_index = local_volume_;
    for (auto& offset : mpi_offsets) {
      // offset is the Cartesian offset within the MPI grid.

      // Compute neighbour rank
      PYQCD_TRACE
      auto num_hops = static_cast<Int>(offset.squaredNorm() + 0.5);
      detail::IVec neighbour_coords = offset + mpi_coord;
      PYQCD_TRACE
      detail::sanitise_coords(neighbour_coords, partition_, num_dims_);
      PYQCD_TRACE
      MPI_Cart_rank(comm, neighbour_coords.data(),
                    &buffer_ranks_[buffer_index]);
      PYQCD_TRACE
      // Here we want to populate buffer_map_, which basically describes how the
      // various buffer indices can be partitioned according to the
      // communication axis and the number of mpi hops.
      detail::IVec buffer_shape = local_shape;
      for (Int dim = 0; dim < num_dims_; ++dim) {
        if (offset[dim] != 0) { // Only interested in cases where comms occurs
          buffer_shape[dim] = static_cast<int>(halo_depth_);
          auto dir
            = (offset[dim] > 0) ? MpiDirection::FRONT : MpiDirection::BACK;
          auto axis = compute_axis(dim, dir);
          PYQCD_TRACE
          buffer_map_[axis][num_hops - 1].push_back(buffer_index);
          PYQCD_TRACE
        }
      }

      buffer_volumes_[buffer_index]
        = detail::compute_volume(buffer_shape.data(),
                                 buffer_shape.data() + num_dims_, 1u);
      PYQCD_TRACE
      // Here we get the site corresponding to the corner of the buffer. The
      // corner is defined as being that where the coordinates are at their
      // smallest (neglecting periodic BCs).
      // TODO: Combine this with previous loop over dimensions
      detail::IVec buffer_corner = unbuffered_region_corner;
      detail::IVec buffered_region_corner = unbuffered_region_corner;
      PYQCD_TRACE
      for (Int dim = 0; dim < num_dims_; ++dim) {
        if (offset[dim] < 0) {
          buffer_corner[dim] -= halo_depth_;
          buffered_region_corner[dim] += local_shape[dim] - halo_depth_;
        }
        else if (offset[dim] > 0) {
          buffer_corner[dim] += local_shape[dim];
        }
      }
      PYQCD_TRACE
      auto buffer_site_iter
        = detail::SiteIterator(buffer_shape.data(),
                               buffer_shape.data() + num_dims_);
      PYQCD_TRACE
      for (auto& site : buffer_site_iter) {
        detail::IVec buffer_site = buffer_corner + detail::site_to_ivec(site);
        PYQCD_TRACE
        Int lexico_index = detail::coords_to_lex_index(buffer_site,
                                                       local_shape_with_halo,
                                                       num_dims_);
        PYQCD_TRACE
        array_indices_local_[lexico_index] = buffer_array_index;
        site_indices_[buffer_array_index++] = lexico_index;
      }
      PYQCD_TRACE

      // Now compute array indices specifying sites that are themselves to be
      // buffered on a neighbouring node
      buffer_site_iter
        = detail::SiteIterator(buffer_shape.data(),
                               buffer_shape.data() + num_dims_);
      buffered_site_indices_[buffer_index].resize(
        buffer_volumes_[buffer_index]);
      Int i = 0;
      for (auto& site : buffer_site_iter) {
        detail::IVec buffered_site
          = buffered_region_corner + detail::site_to_ivec(site);
        Int lexico_index = detail::coords_to_lex_index(buffered_site,
                                                       local_shape, num_dims_);
        buffered_site_indices_[buffer_index][i++] = lexico_index;
      }

      buffer_index++;
    }
  }

#endif

  Int Layout::compute_axis(const Int dimension, const Layout::MpiDirection dir)
  {
    return 2 * dimension + ((dir == MpiDirection::FRONT) ? 1 : 0);
  }
}