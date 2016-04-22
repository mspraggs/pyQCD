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
#include <unordered_map>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <utils/math.hpp>

#ifdef USE_MPI
#include "comms.hpp"
#endif
#include "detail/layout_helpers.hpp"


namespace pyQCD
{
  typedef detail::Int Int;
  typedef detail::Site Site;

  class Layout
  {
  public:
    typedef std::function<Int(const Int)> ArrFunc;

    enum class MpiDirection { FRONT, BACK };

    Layout(const Site& shape, const Site& partition = {},
           const Int halo_depth = 1, const Int max_mpi_hop = 1);
    virtual ~Layout() = default;

    // Functions to retrieve array indices and so on.
    template <typename T,
      typename std::enable_if<not std::is_integral<T>::value>::type* = nullptr>
    Int get_array_index(const T& site) const;
    Int get_array_index(const Int site_index) const
    { return array_indices_[site_index]; }
    inline Int get_site_index(const Int array_index) const
    { return site_indices_[array_index]; }
    inline Site compute_site_coords(const Int site_index) const;
    template <typename T>
    inline void sanitise_site_coords(T& coords) const;

    Int global_volume() const { return global_volume_; }
    const Site& global_shape() const { return global_shape_; }
    int site_mpi_rank(const Int site_index) const
    { return site_mpi_ranks_[site_index]; }
    bool site_is_here(const Int site_index) const
    { return site_is_here_[site_index]; }

    Int local_volume() const { return local_volume_; }
    Int local_size() const { return local_size_; }
    const Site& local_shape() const { return local_shape_; }
    Int num_dims() const { return num_dims_; }
    Int num_buffers() const { return num_buffers_; }
    Int max_mpi_hop() const { return max_mpi_hop_; }

    Int buffer_volume(const Int buffer_index) const
    { return buffer_volumes_[buffer_index]; }
    int buffer_mpi_rank(const Int buffer_index) const
    { return buffer_ranks_[buffer_index]; }
    int surface_mpi_rank(const Int buffer_index) const
    { return surface_ranks_[buffer_index]; }
    const std::vector<Int>& axis_buffer_indices(const Int axis,
                                                const Int mpi_hop) const
    { return axis_hop_buffer_map_[axis][mpi_hop - 1]; }
    const std::vector<Int>& buffer_indices(const Int mpi_hop) const
    { return hop_buffer_map_[mpi_hop - 1]; }
    Int surface_site_corner_index(const Int buffer_index) const
    { return surface_site_corner_indices_[buffer_index]; }
    const std::vector<int>& surface_site_offsets(const Int buffer_index) const
    { return surface_site_offsets_[buffer_index]; }

    static Int compute_axis(const Int dimension, const MpiDirection dir);

  private:
    bool use_mpi_;
    Int num_dims_, local_volume_, local_size_, global_volume_;
    Site global_shape_, local_shape_, local_shape_with_halo_, partition_;
    Site local_corner_;
    // Specifies which dimensions communication is taking place in.
    std::vector<bool> need_comms_;
    // array_indices_[site_index] -> array_index
    std::vector<Int> array_indices_;
    // array_indices_global_[site_index] -> array_index
    std::vector<Int> array_indices_global_;
    // site_indices_[array_index] -> site_index
    std::vector<Int> site_indices_;
    // Specifies rank of node where given (unbuffered) site is located.
    std::vector<int> site_mpi_ranks_;
    // Specifies whether the site is on this node at all (on surface or
    // unbuffered)
    std::vector<bool> site_is_here_;

    int compute_site_mpi_rank(const Int site_index) const;

    /********************************* HALOS ***********************************
     * Here we specify the layout of the halo buffers on this node. The halos
     * are affected by the max_mpi_hop argument in the Layout constructor. This
     * specifies the neighbouring MPI nodes to create buffers for by prescribing
     * the maximum "taxi-driver" or "Manhattan" distance between nodes. For
     * example the taxi-driver distance between nodes specified by Cartesian
     * coordinates (0, 0, 0, 0) and (1, 1, 0, 0) is 2.
     *
     * The reason for describing the buffers in this light is to afford maximum
     * flexibility when initiating communications between nodes. The edges and
     * corners of the node are sometimes needed in computations, and in these
     * cases buffering sites from adjacent nodes may deliver some performance
     * benefit.
     *
     * Each halo buffer is given an index in the Layout constructor, which is
     * then used to refer to the halo buffer throughout the rest of the code. In
     * addition, std::vector objects are used to map the various dimensions to
     * the halo buffers.
     *
     * It's also useful to describe the way we differentiate between "forwards"
     * and "backwards" in the MPI grid. For a given dimension n, we map to the
     * backwards direction using 2 * n, whilst for the forwards direction we
     * map using 2 * n + 1. We describe these values as the axes for the
     * communication. For the sake of convenience, various functions also accept
     * an MpiDirection value (see enum class above)
     */
    // Defines the number of halo buffers
    Int num_buffers_;
    // Defines the number of sites the MPI halo. This is enforced to be less
    // than any of the dimensions in the local_shape_, for the sake
    Int halo_depth_;
    // Maximum MPI hop - just some storage
    Int max_mpi_hop_;
    // Define the mapping between an axis, the max_mpi_hop and the indices of
    // the buffers. The first index will be the axis, whilst the second index is
    // the mpi_hop parameter - 1, both previously mentioned. The inner-most
    // std::vector contains the buffer indices associated with these parameters.
    std::vector<std::vector<std::vector<Int>>> axis_hop_buffer_map_;
    // Define the mapping between the max_mpi_hop and the indices of the
    // buffers. The first index is the mpi_hop parameter - 1. The inner-most
    // std::vector contains the buffer indices associated with these parameters.
    std::vector<std::vector<Int>> hop_buffer_map_;
    // These define the array indices for the sites that belong in the halo of
    // other neighbours. The first index is the buffer index, the second is the
    // lexicographic index of the site within the surface.
    std::vector<std::vector<int>> surface_site_offsets_;
    std::vector<Int> surface_site_corner_indices_;
    // This defines the buffer volumes as the number of lattice sites within the
    // buffer.
    std::vector<Int> buffer_volumes_;
    // Specifies the ranks of the MPI nodes associated with the buffers.
    std::vector<int> buffer_ranks_;
    // Similar to buffer_ranks_, but specifies the mpi rank of the node in the
    // opposite direction within the mpi grid.
    std::vector<int> surface_ranks_;


    // Here we specify some convenience functions for halo operations
    void initialise_buffers(const Int max_mpi_hop);
    detail::IVec compute_buffer_shape(const detail::IVec& offset) const;
    void initialise_unbuffered_sites();
    void handle_offset(const detail::IVec& offset,
                       const detail::IVec& surface_corner,
                       const detail::IVec& buffer_shape,
                       const Int buffer_index);

    template <typename T>
    Site local_to_global_coords(const T& site) const;
  };


  class LexicoLayout : public Layout
  {
  public:
    LexicoLayout(const Site& shape, const Site& partition,
                 const Int halo_depth = 1, const Int max_mpi_hop = 1)
      : Layout(shape, partition, halo_depth, max_mpi_hop)
    { }
  };


  template <typename T,
    typename std::enable_if<not std::is_integral<T>::value>::type*>
  inline Int Layout::get_array_index(const T& site) const
  {
    // Compute the lexicographic index of the specified site and use it to
    // to get the array index (coordinate at site[0] varies slowest; that at
    // site[ndim - 1] varies fastest.
    Int site_index = 0;
    for (Int i = 0; i < num_dims_; ++i) {
      site_index *= global_shape_[i];
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
      ret[i] = mod(site_index_copy, global_shape_[i]);
      site_index_copy /= global_shape_[i];
    }
    return ret;
  }

  template <typename T>
  inline void Layout::sanitise_site_coords(T& coords) const
  {
    // Apply periodic boundary conditions to coords
    for (unsigned i = 0; i < num_dims_; ++i) {
      coords[i] = mod(coords[i], global_shape_[i]);
    }
  }

  template <typename T>
  Site Layout::local_to_global_coords(const T& site) const
  {
    Site ret = local_corner_;
    for (Int dim = 0; dim < num_dims_; ++dim) {
      ret[dim] += static_cast<Int>(site[dim])
                  - (need_comms_[dim] ? halo_depth_ : 0);
    }
    detail::sanitise_coords(ret, global_shape_, num_dims_);
    return ret;
  }
}

#endif