#ifndef MATH_HPP
#define MATH_HPP

/* This file provides basic mathematics functions */

#include <core/types.hpp>


namespace pyQCD
{
  const Real pi = 3.14159265358979323846264338327950288419716939937510;
  const Complex I(0.0, 1.0);

  int mod(const int i, const int n);
}

#endif
