#ifndef LATTICE_HPP
#define LATTICE_HPP

/* This file declares and defines the Lattice class. This is basically an Array
 * but with a Layout member specifying the relationship between the sites and
 * the Array index. In addition, there are operator() implementations to access
 * elements using site coordinates or a lexicographic index.
 */

#include <cassert>
#include <stdexcept>
#include <vector>

#include "detail/lattice_expr.hpp"
#include "layout.hpp"


namespace pyQCD
{
  template <typename T, template <typename> class Alloc = std::allocator>
  class Lattice : public LatticeExpr<Lattice<T>, T>
  {
  public:
    Lattice(const Layout& layout) : layout_(&layout)
    { this->data_.resize(layout.volume()); }
    Lattice(const Layout& layout, const T& val)
      : layout_(&layout), data_(layout.volume(), val)
    {}
    Lattice(const Lattice<T, Alloc>& lattice) = default;
    template <typename U1, typename U2>
    Lattice(const LatticeExpr<U1, U2>& expr)
    {
      this->data_.resize(expr.size());
      for (unsigned long i = 0; i < expr.size(); ++i) {
        this->data_[i] = static_cast<T>(expr[i]);
      }
      layout_ = expr.layout();
    }
    Lattice(Lattice<T, Alloc>&& lattice) = default;

    T& operator[](const int i) { return data_[i]; }
    const T& operator[](const int i) const { return data_[i]; }

    typename std::vector<T>::iterator begin() { return data_.begin(); }
    typename std::vector<T>::const_iterator begin() const
    { return data_.begin(); }
    typename std::vector<T>::iterator end() { return data_.end(); }
    typename std::vector<T>::const_iterator end() const { return data_.end(); }

    template <typename U>
    LatticeView<T, U> slice(const std::vector<int>& slice_spec);

    T& operator()(const int i)
    { return this->data_[layout_->get_array_index(i)]; }
    const T& operator()(const int i) const
    { return this->data_[layout_->get_array_index(i)]; }
    template <typename U>
    T& operator()(const U& site)
    { return this->data_[layout_->get_array_index(site)]; }
    template <typename U>
    const T& operator()(const U& site) const
    { return this->data_[layout_->get_array_index(site)]; }

    Lattice<T, Alloc>& operator=(const Lattice<T, Alloc>& lattice);
    Lattice<T, Alloc>& operator=(Lattice<T, Alloc>&& lattice) = default;
    template <typename U1, typename U2>
    Lattice<T, Alloc>& operator=(const LatticeExpr<U1, U2>& expr)
    {
      pyQCDassert ((this->data_.size() == expr.size()),
                   std::out_of_range("Array::data_"));
      T* ptr = &(this->data_)[0];
      for (unsigned long i = 0; i < expr.size(); ++i) {
        ptr[i] = static_cast<T>(expr[i]);
      }
      layout_ = expr.layout();
      return *this;
    }

    Lattice<T, Alloc>& operator=(const T& rhs)
    {
      data_.assign(data_.size(), rhs);
      return *this;
    }

#define LATTICE_OPERATOR_ASSIGN_DECL(op)				                             \
    template <typename U,                                                    \
      typename std::enable_if<                                               \
		    not std::is_base_of<LatticeObj, U>::value>::type* = nullptr>         \
    Lattice<T, Alloc>& operator op ## =(const U& rhs);	                     \
    template <typename U>                                                    \
    Lattice<T, Alloc>& operator op ## =(const Lattice<U, Alloc>& rhs);

    LATTICE_OPERATOR_ASSIGN_DECL(+);
    LATTICE_OPERATOR_ASSIGN_DECL(-);
    LATTICE_OPERATOR_ASSIGN_DECL(*);
    LATTICE_OPERATOR_ASSIGN_DECL(/);

    unsigned long size() const { return data_.size(); }
    unsigned int volume() const { return layout_->volume(); }
    unsigned int num_dims() const { return layout_->num_dims(); }
    const std::vector<unsigned int>& shape() const
    { return layout_->shape(); }
    const Layout* layout() const { return layout_; }

  protected:
    const Layout* layout_;
    std::vector<T, Alloc<T> > data_;
  };


  template <typename T, template <typename> class Alloc>
  template <typename U>
  LatticeView<T, U> Lattice<T, Alloc>::slice(const std::vector<int>& slice_spec)
  {
    // Creates a LatticeView object that references the slice specified by
    // slice_spec. slice_spec is a vector with length equal to num_dims(),
    // specifying how each dimension should be used in the slice. Positive
    // integers denote a specific coordinate to slice on the given axis, whilst
    // a negative value specifies that all the sites along the given axis should
    // be incorporated into the slice.
    auto test_func = [&] (const Layout::Int index)
    {
      auto index_copy = index;
      const unsigned int size = num_dims();
      for (unsigned int i = 1; i <= size; ++i) {
        auto rem = index_copy % shape()[size - i];
        index_copy /= shape()[size - i];
        if (rem != static_cast<unsigned int>(slice_spec[size - i])
            and slice_spec[size - i] > -1) {
          return false;
        }
      }
      return true;
    };
    return LatticeView<T, U>(*this, std::move(test_func));
  }


  template <typename T, template <typename> class Alloc>
  Lattice<T, Alloc>& Lattice<T, Alloc>::operator=(
    const Lattice<T, Alloc>& lattice)
  {
    if (layout_) {
      pyQCDassert (lattice.volume() == volume(),
        std::invalid_argument("lattice.volume() != volume()"));
    }
    else {
      layout_ = lattice.layout_;
    }
    if (&lattice != this) {
      for (unsigned int i = 0; i < volume(); ++i) {
        (*this)(lattice.layout_->get_site_index(i)) = lattice[i];
      }
    }
    return *this;
  }


#define LATTICE_OPERATOR_ASSIGN_IMPL(op)                                    \
  template <typename T, template <typename> class Alloc>                    \
  template <typename U,                                                     \
    typename std::enable_if<                                                \
      not std::is_base_of<LatticeObj, U>::value>::type*>                    \
  Lattice<T, Alloc>& Lattice<T, Alloc>::operator op ## =(                   \
    const U& rhs)                                                           \
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
  Lattice<T, Alloc>&                                                        \
  Lattice<T, Alloc>::operator op ## =(const Lattice<U, Alloc>& rhs)         \
  {                                                                         \
    pyQCDassert (rhs.size() == data_.size(),                                \
      std::out_of_range("Lattices must be the same size"));                 \
    for (unsigned long i = 0; i < data_.size(); ++i) {                      \
      data_[i] op ## = rhs[i];                                              \
    }                                                                       \
    return *this;                                                           \
  }

LATTICE_OPERATOR_ASSIGN_IMPL(+);
LATTICE_OPERATOR_ASSIGN_IMPL(-);
LATTICE_OPERATOR_ASSIGN_IMPL(*);
LATTICE_OPERATOR_ASSIGN_IMPL(/);
}

#endif
