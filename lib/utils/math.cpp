#include <utils/math.hpp>

/* Implementation of functions in math.cpp */

namespace pyQCD
{
  int mod(const int i, const int n)
  {
    return (i % n + n) % n;
  }
}
