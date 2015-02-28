#ifndef ARRAY_EXPR_HPP
#define ARRAY_EXPR_HPP

/* This file provides expression templates for the Array class, so that
 * temporaries do not need to be created when performing arithmetic operations.
 */

#include <cassert>

#include <typeinfo>
#include <type_traits>

#include <utils/templates.hpp>


namespace pyQCD
{
  template <typename T, template <typename> class Alloc, typename U>
  class Array;

  template <typename T1, typename T2>
  class ArrayExpr;

  template <typename T>
  class ArrayConst;

  template <typename T, template <typename> class Alloc>
  class Lattice;

  // These traits classes allow us to switch between a const ref and simple
  // value in expression subclasses, avoiding returning dangling references.
  template <typename T1, typename T2>
  struct ExprReturnTraits
  {
    typedef T2 type;
  };


  template <typename T1, template <typename> class T2, typename T3>
  struct ExprReturnTraits<Array<T1, T2, T3>, T1>
  {
    typedef T1& type;
  };


  // These traits classes allow us to switch between a const ref and simple
  // value in expression subclasses, avoiding returning dangling references.
  template <typename T>
  struct BinaryOperandTraits
  {
    typedef const T& type;
  };


  template <typename T>
  struct BinaryOperandTraits<ArrayConst<T> >
  {
    typedef ArrayConst<T> type;
  };

  // Traits to check first whether supplied type is a Lattice, then test
  // whether Layouts are the same
  template <typename T1, typename T2>
  struct BinaryLayoutTraits
  {
    static bool check_layout(const T1& lhs, const T2& rhs)
    { return true; }
  };


  template <typename T, template <typename> class A>
  struct BinaryLayoutTraits<Lattice<T, A>, Lattice<T, A> >
  {
    static bool check_layout(const Lattice<T, A>& lhs, const Lattice<T, A>& rhs)
    { return typeid(*(lhs.layout_)) == typeid(*(rhs.layout_)); }
  };


  template <typename T1, typename T2>
  class ArrayExpr
  {
    // This is the main expression class from which all others are derived. It
    // uses CRTP to escape inheritance. Parameter T1 is the expression type
    // and T2 is the fundamental type contained in the Array. This allows
    // expressions to be abstracted to a nested hierarchy of types. When the
    // compiler goes through and does it's thing, the definitions of the
    // operations within these template classes are all spliced together.

  public:
    // CRTP magic - call functions in the Array class
    typename ExprReturnTraits<T1, T2>::type operator[](const int i)
    { return static_cast<T1&>(*this)[i]; }
    const typename ExprReturnTraits<T1, T2>::type operator[](const int i) const
    { return static_cast<const T1&>(*this)[i]; }

    int size() const { return static_cast<const T1&>(*this).size(); }

    operator T1&()
    { return static_cast<T1&>(*this); }
    operator T1 const&() const
    { return static_cast<const T1&>(*this); }
  };


  template <typename T>
  class ArrayConst
    : public ArrayExpr<ArrayConst<T>, T>
  {
    // Expression subclass for const operations
  public:
    ArrayConst(const T& scalar)
      : scalar_(scalar)
    { }
    // Here we denote the actual arithmetic operation.
    const T& operator[](const int i) const
    { return scalar_; }

  private:
    const T& scalar_;
  };


  template <typename T1, typename T2, typename T3, typename T4, typename Op>
  class ArrayBinary
    : public ArrayExpr<ArrayBinary<T1, T2, T3, T4, Op>,
        decltype(Op::apply(T3(), T4()))>
  {
  // Expression subclass for binary operations
  public:
    ArrayBinary(const ArrayExpr<T1, T3>& lhs, const ArrayExpr<T2, T4>& rhs)
      : lhs_(lhs), rhs_(rhs)
    {
#ifndef NDEBUG
      bool layouts_equal = BinaryLayoutTraits<T1, T2>::check_layout(
        static_cast<const T1&>(lhs),
        static_cast<const T2&>(rhs));
      assert (layouts_equal);
#endif
    }
    // Here we denote the actual arithmetic operation.
    const decltype(Op::apply(T3(), T4())) operator[](const int i) const
    { return Op::apply(lhs_[i], rhs_[i]); }

    int size() const { return lhs_.size(); }

  private:
    // The members - the inputs to the binary operation
    typename BinaryOperandTraits<T1>::type lhs_;
    typename BinaryOperandTraits<T2>::type rhs_;
  };


  struct Plus
  {
    template <typename T1, typename T2>
    static decltype(T1() + T2()) apply(const T1& lhs, const T2& rhs)
    { return lhs + rhs; }
  };


  struct Minus
  {
    template <typename T1, typename T2>
    static decltype(T1() - T2()) apply(const T1& lhs, const T2& rhs)
    { return lhs - rhs; }
  };


  struct Multiplies
  {
    template <typename T1, typename T2>
    static decltype(T1() * T2()) apply(const T1& lhs, const T2& rhs)
    { return lhs * rhs; }
  };


  struct Divides
  {
    template <typename T1, typename T2>
    static decltype(T1() / T2()) apply(const T1& lhs, const T2& rhs)
    { return lhs / rhs; }
  };

  // Some macros for the operator overloads, as the code is almost
  // the same in each case. For the scalar multiplies I've used
  // some SFINAE to disable these more generalized functions when
  // a ArrayExpr is used.
#define ARRAY_EXPR_OPERATOR(op, trait)                                \
  template <typename T1, typename T2, typename T3, typename T4>       \
  const ArrayBinary<T1, T2, T3, T4, trait>                            \
  operator op(const ArrayExpr<T1, T3>& lhs,                           \
    const ArrayExpr<T2, T4>& rhs)                                     \
  {                                                                   \
    return ArrayBinary<T1, T2, T3, T4, trait>(lhs, rhs);              \
  }                                                                   \
                                                                      \
                                                                      \
  template <typename T1, typename T2, typename T3,                    \
    typename std::enable_if<                                          \
      !is_instance_of_type_temp<T3, pyQCD::ArrayExpr>::value>::type*  \
      = nullptr>                                                      \
  const ArrayBinary<T1, ArrayConst<T3>, T2, T3, trait>                \
  operator op(const ArrayExpr<T1, T2>& array, const T3& scalar)       \
  {                                                                   \
    return ArrayBinary<T1, ArrayConst<T3>, T2, T3, trait>             \
      (array, ArrayConst<T3>(scalar));                                \
  }

  // This macro is for the + and * operators where the scalar can
  // be either side of the operator.
#define ARRAY_EXPR_OPERATOR_REVERSE_SCALAR(op, trait)                 \
  template <typename T1, typename T2, typename T3,                    \
    typename std::enable_if<                                          \
      !is_instance_of_type_temp<T3, pyQCD::ArrayExpr>::value>::type*  \
      = nullptr>                                                      \
  const ArrayBinary<T1, ArrayConst<T3>, T2, T3, trait>                \
  operator op(const T3& scalar, const ArrayExpr<T1, T2>& array)       \
  {                                                                   \
    return ArrayBinary<T1, ArrayConst<T3>, T2, T3, trait>             \
      (array, ArrayConst<T3>(scalar));                                \
  }


  ARRAY_EXPR_OPERATOR(+, Plus);
  ARRAY_EXPR_OPERATOR_REVERSE_SCALAR(+, Plus);
  ARRAY_EXPR_OPERATOR(-, Minus);
  ARRAY_EXPR_OPERATOR(*, Multiplies);
  ARRAY_EXPR_OPERATOR_REVERSE_SCALAR(*, Multiplies);
  ARRAY_EXPR_OPERATOR(/, Divides);
}

#endif