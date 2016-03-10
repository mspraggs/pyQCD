#ifndef PYQCD_LAYOUT_HELPERS_HPP
#define PYQCD_LAYOUT_HELPERS_HPP
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
 * Created by Matt Spraggs on 13/02/16.
 *
 *
 * [DOCUMENTATION HERE]
 */

#include <iostream>

#include <Eigen/Dense>

#include <utils/math.hpp>


namespace pyQCD
{
  namespace detail
  {
    typedef unsigned int Int;
    typedef std::vector<Int> Site;
    typedef Eigen::VectorXi IVec;
    typedef Eigen::Matrix<Int, -1, 1> UVec;

    auto site_to_ivec(const Site& site)
      -> decltype(UVec::Map(site.data(), site.size()).cast<int>());

    template<typename Iter, typename Elem>
    Elem compute_volume(const Iter& begin, const Iter& end, const Elem init)
    {
      // Compute volume of the specified shape
      return std::accumulate(begin, end, init, std::multiplies<Elem>());
    }

    class SiteIterator
    {
      // Little iterator to loop over all sites on a lattice.
      SiteIterator(const Site& shape, const Site& coord, const Int index);

    public:
      template<typename T>
      SiteIterator(const T& begin, const T& end);

      SiteIterator(const Site& shape);

      bool operator!=(const SiteIterator& other) const
      { return index_ != other.volume_; }

      SiteIterator& operator++();

      const Site& operator*() const { return current_site_; }

      SiteIterator begin()
      { return (index_ == 0) ? *this : SiteIterator(shape_); }
      SiteIterator end()
      { return SiteIterator(shape_, shape_, volume_); }

      Int volume() const { return volume_; }
      Int num_dims() const { return num_dims_; }

    private:
      Int num_dims_;
      Int volume_;
      Int index_;
      Site shape_;
      Site current_site_;
    };

    template<typename T>
    SiteIterator::SiteIterator(const T& begin, const T& end)
      : SiteIterator(Site(begin, end))
    { }


    template<typename T>
    T get_all_sites(SiteIterator& iter);

    template<>
    std::vector<Site> get_all_sites(SiteIterator& iter);

    template<>
    std::vector<IVec> get_all_sites(SiteIterator& iter);

    template<>
    Eigen::MatrixXi get_all_sites(SiteIterator& iter);

    template <typename T, typename U>
    Int coords_to_lex_index(const T& site, const U& shape, const Int ndims)
    {
      // Compute lexicographic index corresponding to the cartesian coordinates
      // of the given site.
      Int site_index = 0;
      for (Int i = 0; i < ndims; ++i) {
        site_index *= shape[i];
        site_index += site[i];
      }
      return site_index;
    }

    template <typename T>
    Site lex_index_to_coords(const Int site_index, const T& shape,
                             const Int ndims)
    {
      // Compute lexicographic index corresponding to the cartesian coordinates
      // of the given site.
      auto site_index_copy = site_index;
      Site ret(ndims);
      for (int i = ndims - 1; i > -1; --i) {
        ret[i] = site_index_copy % shape[i];
        site_index_copy /= shape[i];
      }
      return ret;
    }

    template <typename T, typename U>
    void sanitise_coords(T& coord, const U& shape, const Int ndims)
    {
      // Apply in-place periodic BCs to Cartesian coordinates.
      for (int i = 0; i < static_cast<int>(ndims); ++i) {
        coord[i] = mod(coord[i], shape[i]);
      }
    }

    std::vector<Eigen::VectorXi> generate_mpi_offsets(
      const Int max_mpi_hop, const std::vector<bool>& need_comms);

    IVec compute_first_unbuffered_site(const std::vector<bool>& need_comms,
                                       const Int halo_depth);
  }
}

#endif //PYQCD_LAYOUT_HELPERS_HPP
