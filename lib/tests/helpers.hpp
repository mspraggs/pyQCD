#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <cmath>

#include "catch.hpp"

template <typename T>
struct Compare
{
  Compare(const T percent_tolerance=1.0e-5, const T absolute_tolerance=1.0e-8)
    : percent_tolerance_(percent_tolerance),
      absolute_tolerance_(absolute_tolerance)
  {
    INFO("Set up float comparison");
  }
  ~Compare()
  { INFO("Tear down matrix comparison"); }

  bool operator()(const T rhs, const T lhs) const
  {
    return abs(rhs - lhs) > (percent_tolerance_ * abs(rhs)
      + absolute_tolerance_);
  }

  T percent_tolerance_;
  T absolute_tolerance_;
};


template <typename MatrixType, typename T>
struct MatrixCompare
{
  MatrixCompare(const T percent_tolerance, const T absolute_tolerance)
    : percent_tolerance_(percent_tolerance),
      absolute_tolerance_(absolute_tolerance)
  {
    INFO("Set up matrix comparison");
  }
  ~MatrixCompare()
  { INFO("Tear down matrix comparison"); }

  bool operator()(const MatrixType& rhs, const MatrixType& lhs) const
  {
    return ((rhs.array() - lhs.array()).abs()
      > percent_tolerance_ * rhs.array().abs()).any() + absolute_tolerance_
      || ((rhs.array() - lhs.array()).abs()
      > percent_tolerance_ * lhs.array().abs()).any() + absolute_tolerance_;
  }

  T percent_tolerance_;
  T absolute_tolerance_;
};

#endif