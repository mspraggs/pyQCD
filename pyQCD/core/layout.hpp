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

#ifndef USE_MPI
    Layout(const Site& shape, const ArrFunc& compute_array_index);
#else
    Layout(const Site& shape, const ArrFunc& compute_array_index,
           const Site& partition, const Int halo_depth = 1,
           const Int max_mpi_hop = 1);
#endif
    virtual ~Layout() = default;

    // Functions to retrieve array indices and so on.
    template <typename T,
      typename std::enable_if<not std::is_integral<T>::value>::type* = nullptr>
    Int get_array_index(const T& site) const;
    Int get_array_index(const Int site_index) const
    { return array_indices_local_[site_index]; }
    inline Int get_site_index(const Int array_index) const
    { return site_indices_[array_index]; }
    inline Site compute_site_coords(const Int site_index) const;
    template <typename T>
    inline void sanitise_site_coords(T& coords) const;

    Int local_to_global_array_index(const Int index) const;
    Int global_to_local_array_index(const Int index) const;

    Int volume() const { return local_volume_; }
    Int num_dims() const { return num_dims_; }
    const std::vector<Int>& shape() const
    { return global_shape_; }

  private:
    Int num_dims_, local_volume_, local_size_, global_volume_;
    Site global_shape_, local_shape_, partition_;
    // array_indices_local_[site_index] -> array_index
    std::vector<Int> array_indices_local_;
    // array_indices_global_[site_index] -> array_index
    std::vector<Int> array_indices_global_;
    // site_indices_[array_index] -> site_index
    std::vector<Int> site_indices_;

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
    // Define the mapping between an axis, the max_mpi_hop and the indices of
    // the buffers. The first index will be the axis, whilst the second index is
    // the mpi_hop parameter - 1, both previously mentioned. The inner-most
    // std::vector contains the buffer indices associated with that
    std::vector<std::vector<std::vector<Int>>> buffer_map_;
    // These define the array indices for the halo sites. The first index is the
    // buffer index, the second is the lexicographic index within that buffer.
    std::vector<std::vector<Int>> buffer_indices_;
    // This defines the buffer volumes as the number of lattice sites within the
    // buffer.
    std::vector<Int> buffer_volumes_;
    // Specifies the ranks of the MPI nodes associated with the buffers.
    std::vector<int> buffer_ranks_;


    // Here we specify some convenience functions for halo operations

    // Return -1 if site isn't in a halo
    template <typename T,
      typename std::enable_if<not std::is_integral<T>::value>::type*>
    int get_halo_index(const T& site) const;
    int get_halo_index(const Int site_index) const;

    static Int compute_axis(const Int dimension, const MpiDirection dir);

    void initialise_buffers(const Site& partition, const Int max_mpi_hop);
  };


  class LexicoLayout : public Layout
  {
  public:
#ifndef USE_MPI
    LexicoLayout(const Site& shape)
      : Layout(shape, [] (const Int i) { return i; })
    { }
#else
    LexicoLayout(const Site& shape, const Site& partition,
                 const Int halo_depth = 1, const Int max_mpi_hop = 1)
      : Layout(shape, [] (const Int i) { return i; }, partition, halo_depth,
               max_mpi_hop)
    { }
#endif
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
    return array_indices_local_[site_index];
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
}

#endif