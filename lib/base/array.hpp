#ifndef ARRAY_HPP
#define ARRAY_HPP

/* This file provides an array class, optimized using expression templates and
 * operator overloading. This is probably the most fundamental class in the
 * package, and is the parent to LatticeBase.
 *
 * The reason this has been written from scratch, rather than using the Eigen
 * Array type, is because the Eigen type doesn't support multiplication by
 * generic types, i.e. you can't do something like Array<Matrix3cd> * double.
 *
 * The expression templates that are used by this class can be found in
 * array_expr.hpp
 */

#include <cassert>

#include <type_traits>
#include <vector>

#include <utils/templates.hpp>
#include "array_expr.hpp"


namespace pyQCD
{
  template <typename T1, template <typename> class Alloc, typename T2>
  class Array;

  // Traits to allow passing of inheriting member types to ArrayExpr
  template <typename T1, typename T2, template <typename> class Alloc>
  struct CrtpTrait
  {
    typedef T1 type;
  };


  template <typename T, template <typename> class Alloc>
  struct CrtpTrait<EmptyType, T, Alloc>
  {
    typedef Array<T, Alloc, EmptyType> type;
  };


  template <typename T1, template <typename> class Alloc = std::allocator,
    typename T2 = EmptyType>
  class Array : public ArrayExpr<typename CrtpTrait<T2, T1, Alloc>::type, T1>
  {
  template <typename U1, typename U2, typename U3, typename U4, typename Op>
  friend class ArrayBinary;
  public:
    Array() { }
    Array(const int n, const T1& val) : data_(n, val) { }
    Array(const Array<T1, Alloc, T2>& array) = default;
    Array(Array<T1, Alloc, T2>&& array) = default;
    template <typename U1, typename U2>
    Array(const ArrayExpr<U1, U2>& expr)
    {
      this->data_.resize(expr.size());
      for (int i = 0; i < expr.size(); ++i) {
        this->data_[i] = static_cast<T1>(expr[i]);
      }
    }
    virtual ~Array() = default;

    T1& operator[](const int i) { return data_[i]; }
    const T1& operator[](const int i) const { return data_[i]; }

    typename std::vector<T1>::iterator begin() { return data_.begin(); }
    typename std::vector<T1>::const_iterator begin() const
    { return data_.begin(); }
    typename std::vector<T1>::iterator end() { return data_.end(); }
    typename std::vector<T1>::const_iterator end() const { return data_.end(); }

    Array<T1, Alloc, T2>& operator=(const Array<T1, Alloc, T2>& array) = default;
    Array<T1, Alloc, T2>& operator=(Array<T1, Alloc, T2>&& array) = default;
    Array<T1, Alloc, T2>& operator=(const T1& rhs);

#define ARRAY_OPERATOR_ASSIGN_DECL(op)				                             \
    template <typename U,                                                  \
              typename std::enable_if<                                     \
		!is_instance_of_Array<U, pyQCD::Array>::value>::type* = nullptr>       \
    Array<T1, Alloc, T2>& operator op ## =(const U& rhs);	                 \
    template <typename U>                                                  \
    Array<T1, Alloc, T2>& operator op ## =(const Array<U, Alloc, T2>& rhs);

    ARRAY_OPERATOR_ASSIGN_DECL(+);
    ARRAY_OPERATOR_ASSIGN_DECL(-);
    ARRAY_OPERATOR_ASSIGN_DECL(*);
    ARRAY_OPERATOR_ASSIGN_DECL(/);

    int size() const { return data_.size(); }

  protected:
    std::vector<T1, Alloc<T1> > data_;
  };


  template <typename T1, template <typename> class Alloc, typename T2>
  Array<T1, Alloc, T2>& Array<T1, Alloc, T2>::operator=(const T1& rhs)
  {
    data_.assign(data_.size(), rhs);
  }


#define ARRAY_OPERATOR_ASSIGN_IMPL(op)                                      \
  template <typename T1, template <typename> class Alloc, typename T2>      \
  template <typename U,                                                     \
    typename std::enable_if<                                                \
      !is_instance_of_Array<U, pyQCD::Array>::value>::type*>                \
  Array<T1, Alloc, T2>& Array<T1, Alloc, T2>::operator op ## =(const U& rhs)\
  {                                                                         \
    for (auto& item : data_) {                                              \
      item op ## = rhs;                                                     \
    }                                                                       \
    return *this;                                                           \
  }                                                                         \
                                                                            \
                                                                            \
  template <typename T1, template <typename> class Alloc, typename T2>      \
  template <typename U>                                                     \
  Array<T1, Alloc, T2>&                                                     \
  Array<T1, Alloc, T2>::operator op ## =(const Array<U, Alloc, T2>& rhs)    \
  {                                                                         \
    assert (rhs.size() == data_.size());                                    \
    for (int i = 0; i < data_.size();++i) {                                 \
      data_[i] op ## = rhs[i];                                              \
    }                                                                       \
    return *this;                                                           \
  }

  ARRAY_OPERATOR_ASSIGN_IMPL(+);
  ARRAY_OPERATOR_ASSIGN_IMPL(-);
  ARRAY_OPERATOR_ASSIGN_IMPL(*);
  ARRAY_OPERATOR_ASSIGN_IMPL(/);
}

#endif