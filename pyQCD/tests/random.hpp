#ifndef PYQCD_TEST_RANDOM_HPP
#define PYQCD_TEST_RANDOM_HPP

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
