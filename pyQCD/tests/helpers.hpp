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
    return std::abs(rhs - lhs) < (percent_tolerance_ * std::abs(rhs)
      + absolute_tolerance_)
      and std::abs(rhs - lhs) < (percent_tolerance_ * std::abs(lhs)
      + absolute_tolerance_);
  }

  T percent_tolerance_;
  T absolute_tolerance_;
};


template <typename MatrixType>
struct MatrixCompare
{
  MatrixCompare(const typename MatrixType::RealScalar percent_tolerance,
    const typename MatrixType::RealScalar absolute_tolerance)
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
      < percent_tolerance_ * rhs.array().abs() + absolute_tolerance_).all()
      and ((rhs.array() - lhs.array()).abs()
      < percent_tolerance_ * lhs.array().abs() + absolute_tolerance_).all();
  }

  typename MatrixType::RealScalar percent_tolerance_;
  typename MatrixType::RealScalar absolute_tolerance_;
};


struct TestRandom
{
  TestRandom()
  {
    INFO("Set up random number generator");
    gen = std::mt19937(rd());
    real_dist = std::uniform_real_distribution<>(0, 10);
    int_dist = std::uniform_int_distribution<>(0, 100);
  }
  ~TestRandom() { INFO("Tear down random number generator"); }
  int gen_int() { return int_dist(gen); }
  double gen_real() { return real_dist(gen); }

  std::random_device rd;
  std::mt19937 gen;
  std::uniform_real_distribution<> real_dist;
  std::uniform_int_distribution<> int_dist;
};

#endif