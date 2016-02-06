#ifndef LATTICE_TRAITS_HPP
#define LATTICE_TRAITS_HPP


#include "lattice_expr.hpp"

namespace pyQCD
{
  template <typename T1, typename T2>
  class LatticeExpr;

  template <typename T>
  class LatticeConst;

  template <typename T>
  class Lattice;

  template <typename T>
  class SiteView;

  // These traits classes allow us to switch between a const ref and simple
  // value in expression subclasses, avoiding returning dangling references.
  template <typename T1, typename T2>
  struct ExprReturnTraits
  {
    typedef T2 type;
  };


  template <typename T>
  struct ExprReturnTraits<Lattice<T>, T>
  {
    typedef T& type;
  };


  // These traits classes allow us to switch between a const ref and simple
  // value in expression subclasses, avoiding returning dangling references.
  template <typename T>
  struct OperandTraits
  {
    typedef const T& type;
  };


  template <typename T>
  struct OperandTraits<LatticeConst<T>>
  {
    typedef LatticeConst<T> type;
  };


  template <typename T1, typename T2>
  struct BinaryOperandTraits
  {
    static unsigned long size(const T1& lhs, const T2& rhs)
    { return lhs.size(); }
    static bool equal_size(const T1& lhs, const T2& rhs)
    { return lhs.size() == rhs.size(); }
    static const Layout& layout(const T1& lhs, const T2& rhs)
    { return lhs.layout(); }
    static Int site_size(const T1& lhs, const T2& rhs)
    { return lhs.site_size(); }
    static bool equal_layout(const T1& lhs, const T2& rhs)
    {
      auto lhs_ptr = &lhs.layout();
      auto rhs_ptr = &rhs.layout();
      if (rhs_ptr != nullptr and lhs_ptr != nullptr) {
        return typeid(*rhs_ptr) == typeid(*lhs_ptr)
          and lhs.site_size() == rhs.site_size();
      }
      else {
        return true;
      }
    }
  };


  template <typename T1, typename T2>
  struct BinaryOperandTraits<T1, LatticeConst<T2>>
  {
    static unsigned long size(const T1& lhs, const LatticeConst<T2>& rhs)
    { return lhs.size(); }
    static bool equal_size(const T1& lhs, const LatticeConst<T2>& rhs)
    { return true; }
    static const Layout& layout(const T1& lhs, const LatticeConst<T2>& rhs)
    { return lhs.layout(); }
    static Int site_size(const T1& lhs, const LatticeConst<T2>& rhs)
    { return lhs.site_size(); }
    static bool equal_layout(const T1& lhs, const LatticeConst<T2>& rhs)
    { return true; }
  };


  template <typename T1, typename T2>
  struct BinaryOperandTraits<LatticeConst<T1>, T2>
  {
    static unsigned long size(const LatticeConst<T1>& lhs, const T2& rhs)
    { return rhs.size(); }
    static bool equal_size(const LatticeConst<T1>& lhs, const T2& rhs)
    { return true; }
    static const Layout& layout(const LatticeConst<T1>& lhs, const T2& rhs)
    { return rhs.layout(); }
    static Int site_size(const LatticeConst<T1>& lhs, const T2& rhs)
    { return rhs.site_size(); }
    static bool equal_layout(const LatticeConst<T1>& lhs, const T2& rhs)
    { return true; }
  };


  template <typename T1, typename T2>
  struct BinaryOperandTraits<SiteView<T1>, SiteView<T2>>
  {
    static unsigned long size(const SiteView<T1>& lhs, const SiteView<T2>& rhs)
    { return lhs.size(); }
    static const Layout& layout(const SiteView<T1>& lhs,
                                const SiteView<T2>& rhs)
    { return rhs.layout(); }
    static bool equal_size(const SiteView<T1>& lhs, const SiteView<T2>& rhs)
    { return true; }
    static bool equal_layout(const SiteView<T1>& lhs, const SiteView<T2>& rhs)
    { return true; }
  };
}

#endif