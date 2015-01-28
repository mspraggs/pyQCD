#ifndef ARRAY_EXPR_HPP
#define ARRAY_EXPR_HPP

/* This file provides expression templates for the Array class, so that
 * temporaries do not need to be created when performing arithmetic operations.
 */


namespace pyQCD
{
  template <typename T, template <typename> class Alloc>
  class Array;

  template <typename T1, typename T2>
  class ArrayExpr;

  template <typename T>
  class ArrayConst;

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
    T2& operator[](const int i)
    { return static_cast<T1&>(*this)[i]; }
    const T2& operator[](const int i)
    { return static_cast<T1&>(*this)[i]; }

    int size() const { return static_cast<const Array&>(*this).size(); }

    operator T1&()
    { return static_cast<T1&>(*this); }
    operator T1 const&() const
    { return static_cast<const T1&>(*this); }
  };

  // These traits classes allow us to switch between a const ref and simple
  // value in expression subclasses, avoiding memory issues.
  template <typename T>
  class BinaryTraits
  {
  public:
    typedef const T& type;
  };


  template <typename T>
  class BinaryTraits<ArrayConst<T> >
  {
  public:
    typedef ArrayConst<T> type;
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
    { }
    // Here we denote the actual arithmetic operation.
    decltype(Op::apply(T3(), T4()))& operator[](const int i) const
    { return Op::apply(lhs_[i], rhs_[i]); }

    int size() const { return lhs_.size(); }

  private:
    // The members - the inputs to the binary operation
    typename BinaryTraits<T1>::type rhs_;
    typename BinaryTraits<T2>::type lhs_;
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


#define ARRAY_EXPR_OPERATOR(op, trait)                          \
  template <typename T1, typename T2, typename T3, typename T4> \
  const ArrayBinary<T1, T2, T3, T4, trait>                      \
  operator op(const ArrayExpr<T1, T3>& lhs,                     \
    const ArrayExpr<T2, T4>& rhs)                               \
  {                                                             \
    return ArrayBinary<T1, T2, T3, T4, trait>(lhs, rhs);        \
  }                                                             \
                                                                \
                                                                \
  template <typename T1, typename T2, typename T3,              \
    typename std::enable_if<                                    \
      !is_instance_of_type_temp<T3, ArrayExpr>::value>::type*   \
  const ArrayBinary<T1, ArrayConst<T3>, T2, T3, trait>          \
  operator op(const ArrayExpr<T1, T2>& array, const T3& scalar) \
  {                                                             \
    return ArrayBinary<T1, ArrayConst<T3>, T2, T3, trait>       \
      (array, ArrayConst<T3>(array));                           \
  }


#define ARRAY_EXPR_OPERATOR_REVERSE_SCALAR(op, trait)           \
  template <typename T1, typename T2, typename T3,              \
    typename std::enable_if<                                    \
      !is_instance_of_type_temp<T3, ArrayExpr>::value>::type*   \
  const ArrayBinary<T1, ArrayConst<T3>, T2, T3, trait>          \
  operator op(const T3& scalar, const ArrayExpr<T1, T2>& array) \
  {                                                             \
    return ArrayBinary<T1, ArrayConst<T3>, T2, T3, trait>       \
      (array, ArrayConst<T3>(array));                           \
  }
}


  ARRAY_EXPR_OPERATOR(+, Plus);
  ARRAY_EXPR_OPERATOR_REVERSE_SCALAR(+, Plus);
  ARRAY_EXPR_OPERATOR(-, Minus);
  ARRAY_EXPR_OPERATOR(*, Multiplies);
  ARRAY_EXPR_OPERATOR_REVERSE_SCALAR(*, Multiplies);
  ARRAY_EXPR_OPERATOR(/, Divides);

#endif