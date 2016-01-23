#ifndef RANDOM_HPP
#define RANDOM_HPP

/* This file contains the random number generator used by the rest of the code.
 */

#include <random>
#include <vector>


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