#ifndef PYQCD_LATTICE_EXPR_HPP
#define PYQCD_LATTICE_EXPR_HPP

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
 * This file provides expression templates for the Lattice class, so that
 * temporaries do not need to be created when performing arithmetic operations.
 */

#include <memory>
#include <typeinfo>
#include <type_traits>

#include <utils/macros.hpp>

#include "../layout.hpp"


namespace pyQCD
{
  class LatticeObj { };

  namespace detail
  {
    template <int... Ints>
    struct Seq {};

    template <int Size, int... Ints>
    struct SeqGen : SeqGen<Size - 1, Size - 1, Ints...> {};

    template <int... Ints>
    struct SeqGen<1, Ints...>
    {
      typedef Seq<Ints...> type;
    };
  }


  template <typename Op, typename... Vals>
  class LatticeExpr : public std::tuple<Op, Vals...>, LatticeObj
  {
  public:
    using std::tuple<Op, Vals...>::tuple;
  };


  template <typename T,
    typename std::enable_if<std::is_base_of<LatticeObj, T>::value>::type*
    = nullptr>
  auto eval(const unsigned int i, const T& lattice_obj)
    -> decltype(lattice_obj[i])
  {
    return lattice_obj[i];
  }


  template <typename Op, typename... Vals, int... Ints>
  auto eval(const unsigned int i, const LatticeExpr<Op, Vals...>& expr,
            const detail::Seq<Ints...>)
    -> decltype(std::get<0>(expr).eval(eval(i, std::get<Ints>(expr))...))
  {
    return std::get<0>(expr).eval(eval(i, std::get<Ints>(expr))...);
  }


  template <typename Op, typename... Vals>
  auto eval(const unsigned int i, const LatticeExpr<Op, Vals...>& expr)
    -> decltype(
      eval(i, expr, typename detail::SeqGen<sizeof...(Vals) + 1>::type()))
  {
    return eval(i, expr, typename detail::SeqGen<sizeof...(Vals) + 1>::type());
  }


  template <typename T>
  class LatticeConst : LatticeObj
  {
  public:
    LatticeConst(const T& value) : value_(value) {}

    const T& operator[](const int) const { return value_; }
  private:
    const T value_;
  };


  template <typename T, typename U>
  struct Add
  {
    static auto eval(const T& op1, const U& op2) -> decltype(op1 + op2)
    { return op1 + op2; }
  };


  template <typename T, typename U>
  struct Sub
  {
    static auto eval(const T& op1, const U& op2) -> decltype(op1 - op2)
    { return op1 - op2; }
  };


  template <typename T, typename U>
  struct Mul
  {
    static auto eval(const T& op1, const U& op2) -> decltype(op1 * op2)
    { return op1 * op2; }
  };


  template <typename T, typename U>
  struct Div
  {
    static auto eval(const T& op1, const U& op2) -> decltype(op1 / op2)
    { return op1 / op2; }
  };


#define PYQCD_EXPR_OP_OVERLOAD(op, functor)\
  template <typename T, typename U,\
    typename T_ = typename std::conditional<\
      std::is_base_of<LatticeObj, T>::value, T, LatticeConst<T>>::type,\
    typename U_ = typename std::conditional<\
      std::is_base_of<LatticeObj, U>::value, U, LatticeConst<U>>::type,\
    typename T__ = decltype(eval(0, std::declval<T_>())),\
    typename U__ = decltype(eval(0, std::declval<U_>())),\
    typename std::enable_if<std::is_base_of<LatticeObj, T>::value or\
                            std::is_base_of<LatticeObj, U>::value, char>::type*\
      = nullptr>\
  auto operator op(const T& op1, const U& op2)\
    -> LatticeExpr<functor<T__, U__>, T_, U_>\
  {\
    return LatticeExpr<functor<T__, U__>, T_, U_>(\
      functor<T__, U__>(), T_(op1), U_(op2));\
  }


  PYQCD_EXPR_OP_OVERLOAD(+, Add)
  PYQCD_EXPR_OP_OVERLOAD(-, Sub)
  PYQCD_EXPR_OP_OVERLOAD(*, Mul)
  PYQCD_EXPR_OP_OVERLOAD(/, Div)
}

#endif