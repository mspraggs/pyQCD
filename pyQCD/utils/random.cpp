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
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 *
 * Created by Matt Spraggs on 10/02/16.
 */

#include <unordered_map>
#include <vector>

#include "random.hpp"


namespace pyQCD {

  int get_thread_num()
  {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
  }

  int get_num_threads()
  {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
  }


  RandomWrapper::RandomWrapper(const std::size_t num_rngs)
  {
    // Initialise the specified number of RandGenerators
    rngs_.reserve(num_rngs);

    std::random_device rd;
    for (std::size_t i = 0; i < num_rngs; ++i) {
      rngs_.emplace_back(rd());
    }
  }

  RandomWrapper& RandomWrapper::instance(const Layout& layout)
  {
    // RandomWrapper should be a singleton.
    static std::unordered_map<unsigned int, std::size_t> map;
    static std::vector<RandomWrapper> rngs;

    // Create a lattice-shape specific hash
    const auto& shape = layout.shape();
    auto hash = static_cast<unsigned int>(shape.size());
    for (auto extent : shape) {
      hash ^= extent + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    if (map.count(hash) == 0) {
      rngs.push_back(RandomWrapper(layout.volume()));
      map[hash] = rngs.size() - 1;
    }

    return rngs[map[hash]];
  }

  void RandomWrapper::set_seeds(const std::vector<std::size_t>& seeds)
  {
    for (std::size_t i = 0; i < rngs_.size(); ++i) {
      rngs_[i].set_seed(seeds.at(i));
    }
  }


  RandomWrapper& rng(const Layout& layout)
  {
    return RandomWrapper::instance(layout);
  }
}