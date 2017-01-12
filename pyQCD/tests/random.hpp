#ifndef PYQCD_TEST_RANDOM_HPP
#define PYQCD_TEST_RANDOM_HPP
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
 * Random number generator for the tests.
 */

#include <random>

#include <boost/test/unit_test.hpp>

struct TestRandom
{
  TestRandom()
  {
    BOOST_TEST_MESSAGE("Set up random number generator");
    gen = std::mt19937(rd());
    real_dist = std::uniform_real_distribution<>(0, 10);
    int_dist = std::uniform_int_distribution<>(0, 100);
  }
  ~TestRandom() { BOOST_TEST_MESSAGE("Tear down random number generator"); }
  int gen_int() { return int_dist(gen); }
  double gen_real() { return real_dist(gen); }

  std::random_device rd;
  std::mt19937 gen;
  std::uniform_real_distribution<> real_dist;
  std::uniform_int_distribution<> int_dist;
};

#endif
