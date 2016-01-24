/* Tests for the Random class */
#define CATCH_CONFIG_MAIN

#include <utils/random.hpp>

#include "helpers.hpp"


TEST_CASE("Testing RNG")
{

  Compare<double> comp(1.0e-3, 1.0e-3);

  unsigned num_trials = 100000;
  double mean = 0.0;

  for (unsigned i = 0; i < num_trials; ++i) {
    auto random = pyQCD::rng().generate_real<double>(0.0, 1.0);
    mean += random;

    REQUIRE((random >= 0.0 and random < 1.0));
  }
  mean /= num_trials;

  REQUIRE(comp(mean, 0.5));
}