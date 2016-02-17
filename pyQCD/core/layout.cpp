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
  Layout::Layout(const Site& shape, const ArrFunc& compute_array_index,
                 const Site& partition, const Int halo_depth,
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
      pyQCDassert((local_shape_[i] > max_mpi_hop),
                    std::logic_error("Supplied max_mpi_hop must be smaller "
                                     "than all local shape extents"))
    }

    global_volume_ = detail::compute_volume(shape.begin(), num_dims_);
    local_volume_ = detail::compute_volume(local_shape_.begin(), num_dims_);

    //---------------------- Get some basic MPI info ---------------------------
    initialise_buffers(partition, max_mpi_hop);
  }

  void Layout::initialise_buffers(const Site& partition, const Int max_mpi_hop)
  {
    Site local_shape_with_halo(num_dims_);
    auto halo_depth = halo_depth_;
    std::transform(local_shape_.begin(), local_shape_.end(),
                   local_shape_with_halo.begin(),
                   [&halo_depth] (const Int index)
                   { return index + 2 * halo_depth; });
    Int local_volume_with_halo
      = detail::compute_volume(local_shape_with_halo.begin(), num_dims_);
    array_indices_local_.resize(local_volume_with_halo);

    std::vector<bool> need_comms(num_dims_, true);
    for (Int i = 0; i < num_dims_; ++i) {
      need_comms[i] = partition[i] > 1;
    }

    std::array<MpiDirection, 2> directions{MpiDirection::BACK,
                                           MpiDirection::FRONT};
    detail::IVec mpi_coord(num_dims_);
    MPI_Cart_coords(Communicator::instance().comm(),
                    Communicator::instance().rank(),
                    static_cast<int>(num_dims_), mpi_coord.data());
    PYQCD_TRACE
    detail::IVec local_shape =
      detail::UVec::Map(local_shape_.data(), num_dims_).cast<int>();
    PYQCD_TRACE
    detail::IVec local_origin
      = (local_shape.array() * mpi_coord.array()).matrix();
    PYQCD_TRACE

    auto mpi_offsets = detail::generate_mpi_offsets(max_mpi_hop, need_comms);
    PYQCD_TRACE

    buffer_ranks_.resize(mpi_offsets.size());
    buffer_indices_.resize(mpi_offsets.size());

    // Strap yourself in, things are about to get ugly...
    // TODO: Clear this up/optimize as much as possible.
    buffer_map_.resize(2 * num_dims_);
    for (Int dim = 0; dim < num_dims_; ++dim) {
      auto axis = compute_axis(dim, MpiDirection::FRONT);
      buffer_map_[axis].resize(max_mpi_hop);
    }

    // Now loop through the various MPI offsets and compute the neighbour ranks
    Int buffer_index = 0;
    auto& comm = Communicator::instance().comm();
    PYQCD_TRACE

    for (Int hop = 1; hop < max_mpi_hop + 1; ++hop) {
      auto offset_checker = [hop] (const detail::IVec& site)
      {
        return static_cast<Int>(site.squaredNorm() + 0.5) == hop;
      };

      std::vector<detail::IVec> hop_mpi_offsets;
      // TODO: Optimize this somehow.
      std::copy_if(mpi_offsets.begin(), mpi_offsets.end(),
                   hop_mpi_offsets.begin(), offset_checker);
      num_buffers_ += hop_mpi_offsets.size();
      PYQCD_TRACE

      // TODO: Optimize to remove push_back usage.
      for (auto& mpi_offset : hop_mpi_offsets) {
        detail::IVec neighbour_coords = mpi_offset + mpi_coord;
        detail::sanitise_coords(neighbour_coords, partition_, num_dims_);
        MPI_Cart_rank(comm, mpi_offset.data(), &buffer_ranks_[buffer_index]);

        detail::IVec buffer_shape
          = detail::UVec::Map(local_shape_.data(), num_dims_).cast<int>();
        // Compute buffer shape.
        for (Int dim = 0; dim < num_dims_; ++dim) {
          if (mpi_offset[dim] == 0) {
            continue;
          }
          buffer_shape[dim] = static_cast<int>(2 * halo_depth_);
          auto dir
            = (mpi_offset[dim] > 0) ? MpiDirection::FRONT : MpiDirection::BACK;
          auto axis = compute_axis(dim, dir);
          buffer_map_[axis][hop - 1].push_back(buffer_index);
        }
        PYQCD_TRACE
        Int buffer_volume
          = detail::compute_volume(buffer_shape.data(), num_dims_);
        buffer_volumes_[buffer_index] = buffer_volume;

        detail::SiteIterator buffer_site_iter(buffer_shape.data(),
                                              num_dims_);
        auto buffer_site_array
          = detail::get_all_sites<Eigen::MatrixXi>(buffer_site_iter);
        detail::IVec buffer_origin(num_dims_, halo_depth_);

        for (Int j = 0; j < num_dims_; ++j) {
          if (mpi_offset[j] < 0) {
            buffer_origin[j] = 0;
          }
          else if (mpi_offset[j] > 0) {
            buffer_origin[j] += local_shape_[j];
          }
        }
        PYQCD_TRACE

        buffer_site_array.colwise() += buffer_origin;
        for (Int j = 0; j < buffer_site_array.cols(); ++j) {
          Int array_index = buffer_volume * buffer_index + j;
          Int lexico_index
            = detail::coords_to_lex_index(buffer_site_array.col(j),
                                          local_shape_with_halo, num_dims_);
          array_indices_local_[lexico_index] = array_index + local_volume_;
          site_indices_[array_index + local_volume_] = lexico_index;
        }
        PYQCD_TRACE

        buffer_index++;
      }
    }
  }

#endif
  Int Layout::local_to_global_array_index(const Int index) const
  {
    // Move from local to global array index
    auto site_index = site_indices_[index];
    auto site = compute_site_coords(site_index);
    return 0;
  }

  Int Layout::compute_axis(const Int dimension, const Layout::MpiDirection dir)
  {
    return 2 * dimension + (dir == MpiDirection::FRONT) ? 1 : 0;
  }
}