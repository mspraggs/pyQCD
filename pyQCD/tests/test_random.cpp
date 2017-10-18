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
 * Tests for the Random class.
 */

#include <utils/random.hpp>

#include "helpers.hpp"


TEST_CASE("Testing RNG")
{
  pyQCD::RandGenerator rng(0);

  const Compare<double> comp(1.0e-3, 1.0e-3);

  const unsigned int num_trials = 1000;
  double mean = 0.0;

  for (unsigned int i = 0; i < num_trials; ++i) {
    const auto random = rng.generate_real<double>(0.0, 1.0);
    mean += random;

    REQUIRE((random >= 0.0 and random < 1.0));
  }
  mean /= num_trials;

  REQUIRE(mean == Approx(0.5076074826010093));
}