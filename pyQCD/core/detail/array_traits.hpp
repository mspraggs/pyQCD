#ifndef ARRAY_TRAITS_HPP
#define ARRAY_TRAITS_HPP


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
  struct OperandTraits
  {
    typedef const T& type;
  };


  template <typename T>
  struct OperandTraits<ArrayConst<T> >
  {
    typedef ArrayConst<T> type;
  };

  // Traits to check first whether supplied type is a Lattice, then get the
  // layout from the supplied object, if applicable
  template <typename T>
  struct CrtpLayoutTraits
  {
    static const Layout* get_layout(const T& arr)
    { return arr.layout(); }
  };


  template <typename T1, template <typename> class A, typename T2>
  struct CrtpLayoutTraits<Array<T1, A, T2> >
  {
    static const Layout* get_layout(const Array<T1, A, T2>& lat)
    { return nullptr; }
  };

  // Traits to allow passing of inheriting member types to ArrayExpr
  template <typename T1, typename T2, template <typename> class Alloc>
  struct ArrayCrtpTrait
  {
    typedef T1 type;
  };


  template <typename T, template <typename> class Alloc>
  struct ArrayCrtpTrait<EmptyType, T, Alloc>
  {
    typedef Array<T, Alloc, EmptyType> type;
  };


  template <typename T1, typename T2>
  struct BinaryOperandTraits
  {
    static unsigned long size(const T1& lhs, const T2& rhs)
    { return lhs.size(); }
    static bool equal_size(const T1& lhs, const T2& rhs)
    { return lhs.size() == rhs.size(); }
    static const Layout* layout(const T1& lhs, const T2& rhs)
    { return CrtpLayoutTraits<T1>::get_layout(lhs); }
    static bool equal_layout(const T1& lhs, const T2& rhs)
    {
      const Layout* layout_rhs = CrtpLayoutTraits<T1>::get_layout(lhs);
      const Layout* layout_lhs = CrtpLayoutTraits<T2>::get_layout(rhs);
      if (layout_rhs != nullptr and layout_lhs != nullptr) {
        return typeid(*layout_rhs) == typeid(*layout_lhs);
      }
      else {
        return true;
      }
    }
  };


  template <typename T1, typename T2>
  struct BinaryOperandTraits<T1, ArrayConst<T2> >
  {
    static unsigned long size(const T1& lhs, const ArrayConst<T2>& rhs)
    { return lhs.size(); }
    static bool equal_size(const T1& lhs, const ArrayConst<T2>& rhs)
    { return true; }
    static const Layout* layout(const T1& lhs, const ArrayConst<T2>& rhs)
    { return CrtpLayoutTraits<T1>::get_layout(lhs); }
    static bool equal_layout(const T1& lhs, const ArrayConst<T2>& rhs)
    { return true; }
  };


  template <typename T1, typename T2>
  struct BinaryOperandTraits<ArrayConst<T1>, T2>
  {
    static unsigned long size(const ArrayConst<T1>& lhs, const T2& rhs)
    { return rhs.size(); }
    static bool equal_size(const ArrayConst<T1>& lhs, const T2& rhs)
    { return true; }
    static const Layout* layout(const ArrayConst<T1>& lhs, const T2& rhs)
    { return CrtpLayoutTraits<T2>::get_layout(rhs); }
    static bool equal_layout(const ArrayConst<T1>& lhs, const T2& rhs)
    { return true; }
  };
}

#endif