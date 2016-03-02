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
#include "layout_helpers.hpp"

namespace pyQCD
{
  namespace detail
  {
    auto site_to_ivec(const Site& site)
      -> decltype(UVec::Map(site.data(), site.size()).cast<int>())
    {
      return UVec::Map(site.data(), site.size()).cast<int>();
    }

    SiteIterator::SiteIterator(const Site& shape, const Site& coord,
                               const Int index)
      : num_dims_(static_cast<Int>(shape.size())), index_(index), shape_(shape),
        current_site_(coord)
    {
      volume_ = compute_volume(shape.begin(), shape.end(), 1u);
    }

    SiteIterator::SiteIterator(const Site& shape)
      : SiteIterator(shape, Site(shape.size(), 0), 0)
    { }

    SiteIterator& SiteIterator::operator++()
    {
      // Increment by one lexicographic site (last site coordinate varies
      // fastest).
      Int i = num_dims_ - 1;
      current_site_[i]++;
      index_++;
      while (current_site_[i] == shape_[i] and i != 0) {
        current_site_[i--] = 0;
        current_site_[i]++;
      }
      return *this;
    }

    template<>
    std::vector<Site> get_all_sites(SiteIterator& iter)
    {
      // Generate a std::vector containing all the sites in the iterator
      std::vector <Site> ret(iter.volume(), Site(iter.num_dims()));
      for (Int i = 0; i < iter.volume(); ++i) {
        ret[i] = *iter;
        ++iter;
      }
      return ret;
    }

    template<>
    std::vector<IVec> get_all_sites(SiteIterator& iter)
    {
      // Generate a vector of Eigen vectors from the sites in the iterator
      typedef Eigen::VectorXi IVec;
      typedef Eigen::Matrix<Int, -1, 1> UVec;
      std::vector <IVec> ret(iter.volume(), IVec::Zero(iter.num_dims()));
      for (Int i = 0; i < iter.volume(); ++i) {
        auto& site = *iter;
        ret[i] = UVec::Map(site.data(), site.size()).cast<int>();
        ++iter;
      }
      return ret;
    }

    template<>
    Eigen::MatrixXi get_all_sites(SiteIterator& iter)
    {
      // Generate a vector of Eigen vectors from the sites in the iterator
      typedef Eigen::Matrix<Int, -1, 1> UVec;
      Eigen::MatrixXi ret(iter.volume(), iter.num_dims());
      for (Int i = 0; i < iter.volume(); ++i) {
        auto& site = *iter;
        ret.col(i) = UVec::Map(site.data(), site.size()).cast<int>();
        ++iter;
      }
      return ret;
    }

    std::vector<IVec> generate_mpi_offsets(
      const Int max_mpi_hop, const std::vector<bool>& need_comms)
    {
      // Generate all possible Cartesian coordinate offsets for the specified
      // number of dimensions and maximum number of mpi_hops.

      // First work out how many offsets we actually expect to get. To do this
      // we interpret the number of dimensions that communication needs to
      // occur in (as dictated by need_comms) as the effective dimension of
      // a hypercube, and compute the total number of n-cubes at the boundaries
      // of thise object, where 0 < n <= max_mpi_hop
      Int num_offsets = 0;
      Int ndims = static_cast<Int>(need_comms.size());
      Int effective_ndims = static_cast<Int>(std::count(need_comms.begin(),
                                                        need_comms.end(),
                                                        true));
      for (Int hop = 1; hop < max_mpi_hop + 1; ++hop) {
        // Here we use the formula
        //   2^(n - m) * (n choose m),
        // for the number of m-cubes on the boundaries of the n-cube layout.
        num_offsets
          += choose(effective_ndims, effective_ndims - hop) * std::pow(2, hop);
      }

      Site shape(ndims, 3);
      IVec offset(ndims);
      for (Int i = 0; i < ndims; ++i) {
        shape[i] = (need_comms[i]) ? 3 : 1;
        offset[i] = (need_comms[i]) ? -1 : 0;
      }

      SiteIterator site_iter(shape);
      std::vector<IVec> ret(num_offsets, IVec::Zero(ndims));
      Int i = 0;
      for (auto& site : site_iter) {
        IVec eig_site = UVec::Map(site.data(), ndims).cast<int>() + offset;
        Int hops = static_cast<Int>(eig_site.squaredNorm());
        if (hops > max_mpi_hop or hops == 0) {
          continue;
        }
        ret[i++] = eig_site;
      }

      return ret;
    }
  }
}