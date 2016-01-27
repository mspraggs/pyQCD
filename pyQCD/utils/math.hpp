#ifndef MATH_HPP
#define MATH_HPP

/* This file provides basic mathematics functions */

#include <complex>


namespace pyQCD
{
  const double pi = 3.14159265358979323846264338327950288419716939937510;
  const std::complex<double> I(0.0, 1.0);

  int mod(const int i, const int n);
}

#endif
