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

#include <core/layout.hpp>


// TODO: Adjust this to put an RNG on each site.

namespace pyQCD {
  class RandGenerator
  {
  public:
    RandGenerator() : RandGenerator(std::random_device()()) {}
    RandGenerator(const std::size_t seed) : engine_(seed) {}

    template <typename Real>
    Real generate_real(const Real lower, const Real upper);

    template <typename Int>
    Int generate_int(const Int lower, const Int upper);

    void set_seed(const std::size_t seed) { engine_.seed(seed); }
  private:
    std::mt19937_64 engine_;
  };


  class RandomWrapper
  {
  public:
    static RandomWrapper& instance(const Layout& layout);

    void set_seeds(const std::vector<std::size_t>& seeds);

    RandGenerator& operator[](const std::size_t index) { return rngs_[index]; }
    const RandGenerator& operator[](const std::size_t index) const
    { return rngs_[index]; }

  private:
    RandomWrapper(const std::size_t num_rngs);

    std::vector<RandGenerator> rngs_;
  };


  template<typename Real>
  Real RandGenerator::generate_real(const Real lower, const Real upper)
  {
    std::uniform_real_distribution<Real> dist(lower, upper);
    return dist(engine_);
  }


  template <typename Int>
  Int RandGenerator::generate_int(const Int lower, const Int upper)
  {
    std::uniform_int_distribution<int> dist(lower, upper);
    return dist(engine_);
  }


  RandomWrapper& rng(const Layout& layout);
}

#endif