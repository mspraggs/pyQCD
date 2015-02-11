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
  template <typename T, template <typename> class Alloc = std::allocator>
  class Array : public ArrayExpr<Array<T, Alloc>, T>
  {
  public:
    Array(const int n, const T& val) : data_(n, val) { }
    Array(const Array<T, Alloc>& array);
    Array(Array<T, Alloc>&& array);
    template <typename U1, typename U2>
    Array(const ArrayExpr<U1, U2>& expr)
    {
      this->data_.resize(expr.size());
      for (int i = 0; i < expr.size(); ++i) {
        this->data_[i] = static_cast<T>(expr[i]);
      }
    }
    virtual ~Array() = default;

    T& operator[](const int i) { return data_[i]; }
    const T& operator[](const int i) const { return data_[i]; }

    typename std::vector<T>::iterator begin() { return data_.begin(); }
    typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
    typename std::vector<T>::iterator end() { return data_.end(); }
    typename std::vector<T>::const_iterator end() const { return data_.end(); }

    Array<T, Alloc>& operator=(const Array<T, Alloc>& array);
    Array<T, Alloc>& operator=(Array<T, Alloc>&& array);
    Array<T, Alloc>& operator=(const T& rhs);

#define ARRAY_OPERATOR_ASSIGN_DECL(op)				                        \
    template <typename U,                                             \
              typename std::enable_if<                                \
		!is_instance_of_Array<U, pyQCD::Array>::value>::type* = nullptr>  \
    Array<T, Alloc>& operator op ## =(const U& rhs);	                \
    template <typename U>                                             \
    Array<T, Alloc>& operator op ## =(const Array<U, Alloc>& rhs);

    ARRAY_OPERATOR_ASSIGN_DECL(+);
    ARRAY_OPERATOR_ASSIGN_DECL(-);
    ARRAY_OPERATOR_ASSIGN_DECL(*);
    ARRAY_OPERATOR_ASSIGN_DECL(/);

    int size() const { return data_.size(); }

  protected:
    std::vector<T, Alloc<T> > data_;
  };


  template <typename T, template <typename> class Alloc>
  Array<T, Alloc>::Array(const Array<T, Alloc>& array)
  {
    data_.resize(array.size());
    for (int i = 0; i < array.size(); ++i) {
      data_[i] = array.data_[i];
    }
  }


  template <typename T, template <typename> class Alloc>
  Array<T, Alloc>::Array(Array<T, Alloc>&& array)
    : data_(std::move(array.data_))
  { }


  template <typename T, template <typename> class Alloc>
  Array<T, Alloc>& Array<T, Alloc>::operator=(const Array<T, Alloc>& array)
  {
    if (&array != this) {
      data_.resize(array.size());
      for (int i = 0; i < array.size(); ++i) {
        data_[i] = array.data_[i];
      }
    }
    return *this;
  }


  template <typename T, template <typename> class Alloc>
  Array<T, Alloc>& Array<T, Alloc>::operator=(Array<T, Alloc>&& array)
  {
    data_ = std::move(array.data_);
    return *this;
  }


#define ARRAY_OPERATOR_ASSIGN_IMPL(op)                                      \
  template <typename T, template <typename> class Alloc>                    \
  template <typename U,                                                     \
    typename std::enable_if<                                                \
      !is_instance_of_Array<U, pyQCD::Array>::value>::type*>                \
  Array<T, Alloc>& Array<T, Alloc>::operator op ## =(const U& rhs)          \
  {                                                                         \
    for (auto& item : data_) {                                              \
      item op ## = rhs;                                                     \
    }                                                                       \
    return *this;                                                           \
  }                                                                         \
                                                                            \
                                                                            \
  template <typename T, template <typename> class Alloc>                    \
  template <typename U>                                                     \
  Array<T, Alloc>&                                                          \
  Array<T, Alloc>::operator op ## =(const Array<U, Alloc>& rhs)             \
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