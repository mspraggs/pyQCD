#ifndef PYQCD_OPERATORS_HPP
#define PYQCD_OPERATORS_HPP
/*
 * This file is part of pyQCD.
 *
 * pyQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
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
 * Defines operators for use in Array expression classes
 */


struct Plus
{
  template <typename T1, typename T2>
  static auto apply(const T1& lhs, const T2& rhs) -> decltype(lhs + rhs)
  { return lhs + rhs; }
};


struct Minus
{
  template <typename T1, typename T2>
  static auto apply(const T1& lhs, const T2& rhs) -> decltype(lhs - rhs)
  { return lhs - rhs; }
};


struct Multiplies
{
  template <typename T1, typename T2>
  static auto apply(const T1& lhs, const T2& rhs) -> decltype(lhs * rhs)
  { return lhs * rhs; }
};


struct Divides
{
  template <typename T1, typename T2>
  static auto apply(const T1& lhs, const T2& rhs) -> decltype(lhs / rhs)
  { return lhs / rhs; }
};


struct Adjoint
{
  template <typename T>
  static auto apply(const T& operand) -> decltype(operand.adjoint())
  { return operand.adjoint(); }
};

#endif