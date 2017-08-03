/*
 * This file is part of pyQCD.
 *
 * pyQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pyQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Created by Matt Spraggs on 03/08/17.
 */

#include "layout.hpp"


namespace pyQCD
{
  PartitionCompare::PartitionCompare(const unsigned int minor_offset,
                                     const Layout& layout)
      : layout_(&layout), minor_offset_(minor_offset)
  {
    major_remainders_.resize(2 * minor_offset);
    std::iota(major_remainders_.begin(), major_remainders_.end(),
              2 * minor_offset);

    const auto reserve_size =
        static_cast<unsigned int>(std::pow(minor_offset, layout.num_dims()));

    unsigned int level = layout.num_dims() - 1;
    std::vector<unsigned int> remainders(layout.num_dims(), 0);

    minor_remainders_.resize(reserve_size);
    for (unsigned int i = 0; i < reserve_size; ++i) {
      minor_remainders_[i] = remainders;

      while (level > 0 and remainders[level] == minor_offset - 1) {
        remainders[level] = 0;
        level -= 1;
      }

      remainders[level] += 1;

      if (level < layout.num_dims() - 1) {
        level = layout.num_dims() - 1;
      }
    }
  }


  bool PartitionCompare::operator()(const Int first, const Int second) const
  {
    return compute_reference(first) < compute_reference(second);
  }


  unsigned int PartitionCompare::compute_reference(const Int value) const
  {
    const auto coords = layout_->compute_site_coords(value);

    unsigned int minor_result = 0;

    for (unsigned int i = 0; i < minor_remainders_.size(); ++i) {
      bool good_partition = true;

      for (unsigned int j = 0; j < coords.size(); ++j) {
        good_partition &= coords[j] % minor_offset_ == minor_remainders_[i][j];

        if (!good_partition) {
          break;
        }
      }

      if (good_partition) {
        minor_result = i;
        break;
      }
    }

    const auto minor_size = static_cast<unsigned int>(minor_remainders_.size());
    const auto coord_sum = std::accumulate(coords.begin(), coords.end(), 0u);

    return minor_result + (coord_sum % (2 * minor_offset_)) * minor_size;
  }
}