#ifndef LATTICE_TRAITS_HPP
#define LATTICE_TRAITS_HPP


namespace pyQCD
{
  template <typename T1, typename T2>
  class LatticeExpr;

  template <typename T>
  class LatticeConst;

  template <typename T, template <typename> class Alloc>
  class Lattice;

  // These traits classes allow us to switch between a const ref and simple
  // value in expression subclasses, avoiding returning dangling references.
  template <typename T1, typename T2>
  struct ExprReturnTraits
  {
    typedef T2 type;
  };


  template <typename T1, template <typename> class T2>
  struct ExprReturnTraits<Lattice<T1, T2>, T1>
  {
    typedef T1& type;
  };


  // These traits classes allow us to switch between a const ref and simple
  // value in expression subclasses, avoiding returning dangling references.
  template <typename T>
  struct OperandTraits
  {
    typedef const T& type;
  };


  template <typename T>
  struct OperandTraits<LatticeConst<T> >
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
    static const Layout* layout(const T1& lhs, const T2& rhs)
    { return lhs.layout(); }
    static bool equal_layout(const T1& lhs, const T2& rhs)
    {
      if (rhs.layout() != nullptr and lhs.layout() != nullptr) {
        return typeid(*rhs.layout()) == typeid(*lhs.layout());
      }
      else {
        return true;
      }
    }
  };


  template <typename T1, typename T2>
  struct BinaryOperandTraits<T1, LatticeConst<T2> >
  {
    static unsigned long size(const T1& lhs, const LatticeConst<T2>& rhs)
    { return lhs.size(); }
    static bool equal_size(const T1& lhs, const LatticeConst<T2>& rhs)
    { return true; }
    static const Layout* layout(const T1& lhs, const LatticeConst<T2>& rhs)
    { return lhs.layout(); }
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
    static const Layout* layout(const LatticeConst<T1>& lhs, const T2& rhs)
    { return rhs.layout(); }
    static bool equal_layout(const LatticeConst<T1>& lhs, const T2& rhs)
    { return true; }
  };
}

#endif