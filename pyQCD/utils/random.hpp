#ifndef PYQCD_RANDOM_HPP
#define PYQCD_RANDOM_HPP
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
 *
 *
 * This file contains the random number generator used by the rest of the code.
 */

#include <chrono>
#include <random>
#include <vector>


// TODO: Adjust this to put an RNG on each site.

namespace pyQCD {

  int get_thread_num();
  int get_num_threads();

  class Random
  {
    Random(const size_t num_threads);

  public:
    static Random& instance(const size_t num_threads);

    void set_seed(const size_t seed);

    int generate_int(const int lower, const int upper);

    template <typename Real>
    Real generate_real(const Real lower, const Real upper);

  private:
    size_t num_threads_;
    std::vector<std::ranlux48> engines_;
  };

  template <typename Real>
  Real Random::generate_real(const Real lower, const Real upper)
  {
    int thread = get_thread_num();
    std::uniform_real_distribution<Real> dist(lower, upper);
    return dist(engines_[thread]);
  }


  Random& rng()
  {
    return Random::instance(get_num_threads());
  }
}

#endif