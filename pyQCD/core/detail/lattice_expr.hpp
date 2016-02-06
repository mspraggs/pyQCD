#ifndef LATTICE_EXPR_HPP
#define LATTICE_EXPR_HPP

/* This file provides expression templates for the Lattice class, so that
 * temporaries do not need to be created when performing arithmetic operations.
 */

#include <memory>
#include <typeinfo>
#include <type_traits>

#include <utils/macros.hpp>

#include "../layout.hpp"
#include "lattice_traits.hpp"
#include "operators.hpp"


namespace pyQCD
{
  class LatticeObj { };

  // TODO: Eliminate need for second template parameter
  template <typename T1, typename T2>
  class LatticeExpr : public LatticeObj
  {
    // This is the main expression class from which all others are derived. It
    // uses CRTP to escape inheritance. Parameter T1 is the expression type
    // and T2 is the fundamental type contained in the Lattice. This allows
    // expressions to be abstracted to a nested hierarchy of types. When the
    // compiler goes through and does it's thing, the definitions of the
    // operations within these template classes are all spliced together.

  public:
    // CRTP magic - call functions in the Lattice class
    typename ExprReturnTraits<T1, T2>::type operator[](const int i)
    { return static_cast<T1&>(*this)[i]; }
    const typename ExprReturnTraits<T1, T2>::type operator[](const int i) const
    { return static_cast<const T1&>(*this)[i]; }

    unsigned long size() const { return static_cast<const T1&>(*this).size(); }
    Int site_size() const { return static_cast<const T1&>(*this).site_size(); }
    const Layout& layout() const
    { return static_cast<const T1&>(*this).layout(); }

    operator T1&() { return static_cast<T1&>(*this); }
    operator T1 const&() const { return static_cast<const T1&>(*this); }
  };


  template <typename T>
  class LatticeConst
    : public LatticeExpr<LatticeConst<T>, T>
  {
    // Expression subclass for const operations
  public:
    // Need some SFINAE here to ensure no clash with copy/move constructor
    template <typename std::enable_if<
      not std::is_same<T, LatticeConst<T> >::value>::type* = nullptr>
    LatticeConst(const T& scalar) : scalar_(scalar) { }
    const T& operator[](const unsigned long i) const { return scalar_; }

  private:
    const T& scalar_;
  };


  template <typename T1, typename T2, typename Op>
  class LatticeUnary
    : public LatticeExpr<LatticeUnary<T1, T2, Op>,
        decltype(Op::apply(std::declval<T2>()))>
  {
  public:
    LatticeUnary(const LatticeExpr<T1, T2>& operand) : operand_(operand) { }

    const decltype(Op::apply(std::declval<T2>()))
    operator[](const unsigned int i) const { return Op::apply(operand_[i]); }

    unsigned long size() const { return operand_.size(); }
    const Layout& layout() const { return operand_.layout(); }
    Int site_size() const { return operand_.site_size(); }

  private:
    typename OperandTraits<T1>::type operand_;
  };


  template <typename T1, typename T2, typename T3, typename T4, typename Op>
  class LatticeBinary
    : public LatticeExpr<LatticeBinary<T1, T2, T3, T4, Op>,
        decltype(Op::apply(std::declval<T3>(), std::declval<T4>()))>
  {
  // Expression subclass for binary operations
  public:
    LatticeBinary(const LatticeExpr<T1, T3>& lhs, const LatticeExpr<T2, T4>& rhs)
      : lhs_(lhs), rhs_(rhs)
    {
      pyQCDassert((BinaryOperandTraits<T1, T2>::equal_size(lhs_, rhs_)),
        std::out_of_range("LatticeBinary: lhs.size() != rhs.size()"));
      pyQCDassert((BinaryOperandTraits<T1, T2>::equal_layout(lhs_, rhs_)),
        std::bad_cast());
    }
    // Here we denote the actual arithmetic operation.
    const decltype(Op::apply(std::declval<T3>(), std::declval<T4>()))
    operator[](const unsigned long i) const
    { return Op::apply(lhs_[i], rhs_[i]); }

    unsigned long size() const
    { return BinaryOperandTraits<T1, T2>::size(lhs_, rhs_); }
    const Layout& layout() const
    { return BinaryOperandTraits<T1, T2>::layout(lhs_, rhs_); }
    Int site_size() const
    { return BinaryOperandTraits<T1, T2>::site_size(lhs_, rhs_); }

  private:
    // The members - the inputs to the binary operation
    typename OperandTraits<T1>::type lhs_;
    typename OperandTraits<T2>::type rhs_;
  };


  // This class allows views on existing lattice data to be created, without
  // creating copies.
  template <typename T>
  class SiteView : public LatticeExpr<SiteView<T>, T>
  {
  public:
    template <typename U>
    SiteView(Lattice<T>& lattice, const U& site, const unsigned int size)
    {
      auto& layout = lattice.layout();
      Int elem_index = layout.get_array_index(site);
      references_.resize(size);
      for (Int i = 0; i < references_.size(); ++i) {
        references_[i] = &lattice[elem_index + i];
      }
    }

    template <typename U1, typename U2>
    SiteView(const LatticeExpr<U1, U2>& expr);

    SiteView<T>& operator=(const SiteView<T>& site_view);
    template <typename U1, typename U2>
    SiteView<T>& operator=(const LatticeExpr<U1, U2>& expr);

    unsigned long size() const { return references_.size(); }

    T& operator[](const unsigned int i) { return *(references_[i]); }
    const T& operator[](const unsigned int i) const
    { return *(references_[i]); }

  private:
    std::vector<T*> references_;
  };

  template <typename T>
  template <typename U1, typename U2>
  SiteView<T>::SiteView(const LatticeExpr<U1, U2>& expr)
  {
    pyQCDassert ((references_.size() == expr.size()),
      std::out_of_range("Array::data_"));
    T** ptr = &(references_)[0];
    for (unsigned long i = 0; i < expr.size(); ++i) {
      *(ptr[i]) = static_cast<T>(expr[i]);
    }
  }

  template <typename T>
  SiteView<T>& SiteView<T>::operator=(const pyQCD::SiteView<T>& site_view)
  {
    pyQCDassert ((references_.size() == site_view.size()),
      std::out_of_range("Array::data_"));
    T** ptr = &(references_)[0];
    for (unsigned long i = 0; i < site_view.size(); ++i) {
      *(ptr[i]) = static_cast<T>(site_view[i]);
    }
    return *this;
  }

  template <typename T>
  template <typename U1, typename U2>
  SiteView<T>& SiteView<T>::operator=(const LatticeExpr<U1, U2>& expr)
  {
    pyQCDassert ((references_.size() == expr.size()),
      std::out_of_range("Array::data_"));
    T** ptr = &(references_)[0];
    for (unsigned long i = 0; i < expr.size(); ++i) {
      *(ptr[i]) = static_cast<T>(expr[i]);
    }
    return *this;
  }

  // Some macros for the operator overloads, as the code is almost
  // the same in each case. For the scalar multiplies I've used
  // some SFINAE to disable these more generalized functions when
  // a LatticeExpr is used.
#define LATTICE_EXPR_OPERATOR(op, trait)                              \
  template <typename T1, typename T2, typename T3, typename T4>       \
  const LatticeBinary<T1, T2, T3, T4, trait>                          \
  operator op(const LatticeExpr<T1, T3>& lhs,                         \
    const LatticeExpr<T2, T4>& rhs)                                   \
  {                                                                   \
    return LatticeBinary<T1, T2, T3, T4, trait>(lhs, rhs);            \
  }                                                                   \
                                                                      \
                                                                      \
  template <typename T1, typename T2, typename T3,                    \
    typename std::enable_if<                                          \
      not std::is_base_of<LatticeObj, T3>::value>::type* = nullptr>   \
  const LatticeBinary<T1, LatticeConst<T3>, T2, T3, trait>            \
  operator op(const LatticeExpr<T1, T2>& lattice, const T3& scalar)   \
  {                                                                   \
    return LatticeBinary<T1, LatticeConst<T3>, T2, T3, trait>         \
      (lattice, LatticeConst<T3>(scalar));                            \
  }

  // This macro is for the + and * operators where the scalar can
  // be either side of the operator.
#define LATTICE_EXPR_OPERATOR_REVERSE_SCALAR(op, trait)               \
  template <typename T1, typename T2, typename T3,                    \
    typename std::enable_if<                                          \
      not std::is_base_of<LatticeObj, T1>::value>::type* = nullptr>   \
  const LatticeBinary<LatticeConst<T1>, T2, T1, T3, trait>            \
  operator op(const T1& scalar, const LatticeExpr<T2, T3>& lattice)   \
  {                                                                   \
    return LatticeBinary<LatticeConst<T1>, T2, T1, T3, trait>         \
      (LatticeConst<T1>(scalar), lattice);                            \
  }


  LATTICE_EXPR_OPERATOR(+, Plus);
  LATTICE_EXPR_OPERATOR_REVERSE_SCALAR(+, Plus);
  LATTICE_EXPR_OPERATOR(-, Minus);
  LATTICE_EXPR_OPERATOR(*, Multiplies);
  LATTICE_EXPR_OPERATOR_REVERSE_SCALAR(*, Multiplies);
  LATTICE_EXPR_OPERATOR(/, Divides);
}

#endif