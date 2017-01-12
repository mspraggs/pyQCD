#ifndef PYQCD_LATTICE_EXPR_HPP
#define PYQCD_LATTICE_EXPR_HPP

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
 * This file provides expression templates for the Lattice class, so that
 * temporaries do not need to be created when performing arithmetic operations.
 *
 * The implementation is based largely on that found in Antonin Portelli's
 * LatSim code, see https://github.com/aportelli/LatSim/
 */

#include <memory>
#include <typeinfo>
#include <type_traits>

#include <utils/macros.hpp>


namespace pyQCD
{
  class LatticeObj { };

  namespace detail
  {
    // Here we define a set of simple structs to generate sequenced integer
    // parameter packs, encapsulated within a Seq template type. When passed
    // to functions, such Seq instances allow integer parameter packs to be
    // used by the function.
    template<int... Ints>
    struct Seq
    {
    };

    template<int Size, int... Ints>
    struct SeqGen : SeqGen<Size - 1, Size - 1, Ints...>
    {
    };

    template<int... Ints>
    struct SeqGen<1, Ints...>
    {
      typedef Seq<Ints...> type;
    };


    template<typename Op, typename... Vals>
    class LatticeExpr : public std::tuple<Op, Vals...>, LatticeObj
    {
      // Expressions are encoded within tuples, but they must also inherit from
      // LatticeObj to provide a common type to be recognised by overloaded
      // operators (see below).
    public:
      using std::tuple<Op, Vals...>::tuple;
    };


    // The eval functions below are where the magic happens. In combination they
    // are responsible for applying the arithmetic operations encoded by a
    // LatticeExpr object to the individual set of elements described by the
    // index i within the underlying lattice objects.
    template<typename T,
      typename std::enable_if<std::is_base_of<LatticeObj, T>::value>::type*
      = nullptr>
    auto eval(const unsigned int i, const T& lattice_obj)
    -> decltype(lattice_obj[i])
    {
      // This function extracts an individual element from an indexable object
      // inheriting from LatticeObj.
      //
      // Due to the compiler's name-lookup rules, this function is only applied
      // to indexable objects inheriting from LatticeObj that aren't instances
      // of a LatticeExpr template (e.g. Lattice<T> or ConstWrapper<T>).
      return lattice_obj[i];
    }


    template<typename Op, typename... Vals, int... Ints>
    auto eval(const unsigned int i, const LatticeExpr<Op, Vals...>& expr,
              const detail::Seq<Ints...>)
    -> decltype(std::get<0>(expr).eval(eval(i, std::get<Ints>(expr))...))
    {
      // This function is responsible for extracting the operands required by
      // the operator Op. Some variadic template magic allows each element with
      // index i to be extracted and the results passed to Op::eval. Note that
      // eval here is resolved either to the function above (where the operand
      // type isn't a LatticeExpr template instance) or the function below
      // (where it is an expression template instance).
      return std::get<0>(expr).eval(eval(i, std::get<Ints>(expr))...);
    }


    template<typename Op, typename... Vals>
    auto eval(const unsigned int i, const LatticeExpr<Op, Vals...>& expr)
    -> decltype(
    eval(i, expr, typename detail::SeqGen<sizeof...(Vals) + 1>::type()))
    {
      // This is the main entry-point for evaluating the supplied expression
      // with the given index i. It is also used to evaluate subexpressions
      // referred to within parent expressions.
      return eval(i, expr,
                  typename detail::SeqGen<sizeof...(Vals) + 1>::type());
    }


    template<typename T>
    class LatticeConst : LatticeObj
    {
      // This class is used to encapsulate constant objects so that expressions
      // such as lattice / 2.0 or lattice * mat can be computed efficiently.
    public:
      LatticeConst(const T& value) : value_(value)
      { }

      const T& operator[](const int) const
      { return value_; }

    private:
      const T value_;
    };


    // Below I define operators for the basic arithmetic operations.

    template<typename T, typename U>
    struct Add
    {
      static auto eval(const T& op1, const U& op2) -> decltype(op1 + op2)
      { return op1 + op2; }
    };


    template<typename T, typename U>
    struct Sub
    {
      static auto eval(const T& op1, const U& op2) -> decltype(op1 - op2)
      { return op1 - op2; }
    };


    template<typename T, typename U>
    struct Mul
    {
      static auto eval(const T& op1, const U& op2) -> decltype(op1 * op2)
      { return op1 * op2; }
    };


    template<typename T, typename U>
    struct Div
    {
      static auto eval(const T& op1, const U& op2) -> decltype(op1 / op2)
      { return op1 / op2; }
    };


    // In order to prevent unnecessary copies of LatticeObj objects being
    // created upon construction of a LatticeExpr instance, the LatticeExpr
    // template should contain constant reference types. However, this is
    // problematic in the case where operands should be converted to an instance
    // of ConstWrapper. This latter case would result in references to a
    // temporaries. The ConstCheck template below facilitates checking for
    // operands that must be cast to a ConstWrapper and provides the type they
    // should be cast to. Thus temporary objects are passed by value and
    // dangling references are avoided.

    template<typename T, typename U>
    struct ConstCheck
    {
      typedef typename std::conditional<std::is_same<T, U>::value,
        const T&, U>::type type;
    };
  }


  // Macro to save writing this horrible mess four times over (the extra
  // template arguments were originally to introduced to tidy things up, but
  // it's ended up a mess anyway).
#define PYQCD_EXPR_OP_OVERLOAD(op, op_struct)\
  template <typename T, typename U,\
    typename T_ = typename std::conditional<\
      std::is_base_of<LatticeObj, T>::value, T,\
      detail::LatticeConst<T>>::type,\
    typename U_ = typename std::conditional<\
      std::is_base_of<LatticeObj, U>::value, U,\
      detail::LatticeConst<U>>::type,\
    typename T__ = decltype(detail::eval(0, std::declval<T_>())),\
    typename U__ = decltype(detail::eval(0, std::declval<U_>())),\
    typename std::enable_if<std::is_base_of<LatticeObj, T>::value or\
                            std::is_base_of<LatticeObj, U>::value,\
                            char>::type* = nullptr>\
  auto operator op(const T& op1, const U& op2)\
    -> detail::LatticeExpr<op_struct<T__, U__>,\
                           typename detail::ConstCheck<T, T_>::type,\
                           typename detail::ConstCheck<U, U_>::type>\
  {\
    return detail::LatticeExpr<op_struct<T__, U__>,\
                               typename detail::ConstCheck<T, T_>::type,\
                               typename detail::ConstCheck<U, U_>::type>(\
      op_struct<T__, U__>(),\
      typename detail::ConstCheck<T, T_>::type(op1),\
      typename detail::ConstCheck<U, U_>::type(op2));\
  }


  PYQCD_EXPR_OP_OVERLOAD(+, detail::Add)
  PYQCD_EXPR_OP_OVERLOAD(-, detail::Sub)
  PYQCD_EXPR_OP_OVERLOAD(*, detail::Mul)
  PYQCD_EXPR_OP_OVERLOAD(/, detail::Div)
}

#endif