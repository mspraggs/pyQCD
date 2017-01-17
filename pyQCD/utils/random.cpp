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


  Random::Random(const size_t num_threads) : num_threads_(num_threads)
  {
    // Initialise one generator for each thread
    std::random_device rd;
    engines_.resize(num_threads_);
    for (size_t i = 0; i < num_threads_; ++i) {
      engines_[i] = std::mt19937(static_cast<size_t>(rd()));
    }
  }

  Random& Random::instance(const size_t num_threads)
  {
    // Random should be a singleton.
    static Random ret(num_threads);
    return ret;
  }

  void Random::set_seed(const std::vector<size_t>& seed)
  {
    // Set the seed
    for (size_t i = 0; i < num_threads_; ++i) {
      engines_[i].seed(seed[get_thread_num()]);
    }
  }

  int Random::generate_int(const int lower, const int upper)
  {
    int thread = get_thread_num();
    std::uniform_int_distribution<int> dist(lower, upper);
    return dist(engines_[thread]);
  }


  Random& rng()
  {
    return Random::instance(get_num_threads());
  }
}