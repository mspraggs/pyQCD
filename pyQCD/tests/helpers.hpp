#ifndef PYQCD_HELPERS_HPP
#define PYQCD_HELPERS_HPP
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
 * Helper classes to compare matrices and floating point types.
 */

#include <cmath>
#include <random>

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
    return std::abs(rhs - lhs) <= (percent_tolerance_ * std::abs(rhs)
      + absolute_tolerance_)
      and std::abs(rhs - lhs) <= (percent_tolerance_ * std::abs(lhs)
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
      <= percent_tolerance_ * rhs.array().abs() + absolute_tolerance_).all()
      and ((rhs.array() - lhs.array()).abs()
      <= percent_tolerance_ * lhs.array().abs() + absolute_tolerance_).all();
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