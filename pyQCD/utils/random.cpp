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
    long seed = std::chrono::system_clock::now().time_since_epoch().count();
    engines_.resize(num_threads_);
    for (size_t i = 0; i < num_threads_; ++i) {
      engines_[i] = std::ranlux48(seed + i);
    }
  }

  Random& Random::instance(const size_t num_threads)
  {
    // Random should be a singleton.
    static Random ret(num_threads);
    return ret;
  }

  void Random::set_seed(const size_t seed)
  {
    // Set the seed
    for (size_t i = 0; i < num_threads_; ++i) {
      engines_[i].seed(get_thread_num() + i);
    }
  }

  int Random::generate_int(const int lower, const int upper)
  {
    int thread = get_thread_num();
    std::uniform_int_distribution<int> dist(lower, upper);
    return dist(engines_[thread]);
  }
}