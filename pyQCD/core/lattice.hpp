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
  template <typename T>
  using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

  enum class Partition {EVEN, ODD};


  template <typename T>
  class Lattice : public LatticeExpr<Lattice<T>, T>
  {
  public:
#ifdef MPI_VERSION
#else
    Lattice(const Layout& layout, const Int site_size = 1)
      : site_size_(site_size), layout_(&layout)
    { this->data_.resize(site_size_ * layout.volume()); }
    Lattice(const Layout& layout, const T& val, const Int site_size = 1)
      : site_size_(site_size), layout_(&layout),
        data_(site_size_ * layout.volume(), val)
    {}
#endif
    Lattice(const Lattice<T>& lattice) = default;
    template <typename U1, typename U2>
    Lattice(const LatticeExpr<U1, U2>& expr)
    {
      this->data_.resize(expr.size());
      for (unsigned long i = 0; i < expr.size(); ++i) {
        this->data_[i] = static_cast<T>(expr[i]);
      }
      layout_ = &expr.layout();
      site_size_ = expr.site_size();
    }
    Lattice(Lattice<T>&& lattice) = default;

    T& operator[](const int i) { return data_[i]; }
    const T& operator[](const int i) const { return data_[i]; }

    typename aligned_vector<T>::iterator begin() { return data_.begin(); }
    typename aligned_vector<T>::const_iterator begin() const
    { return data_.begin(); }
    typename aligned_vector<T>::iterator end() { return data_.end(); }
    typename aligned_vector<T>::const_iterator end() const
    { return data_.end(); }

    T& operator()(const Int site, const Int elem = 0)
    { return this->data_[site_size_ * layout_->get_array_index(site) + elem]; }
    const T& operator()(const Int site, const Int elem = 0) const
    { return this->data_[site_size_ * layout_->get_array_index(site) + elem]; }
    template <typename U>
    T& operator()(const U& site, const Int elem = 0)
    { return this->data_[site_size_ * layout_->get_array_index(site) + elem]; }
    template <typename U>
    const T& operator()(const U& site, const Int elem = 0) const
    { return this->data_[site_size_ * layout_->get_array_index(site) + elem]; }

    template <typename U>
    SiteView<T> site_view(const U& site)
    { return SiteView<T>(*this, site, site_size_); }
    SiteView<T> site_view(const Int site)
    { return SiteView<T>(*this, site, site_size_); }

    Lattice<T>& operator=(const Lattice<T>& lattice);
    Lattice<T>& operator=(Lattice<T>&& lattice) = default;
    template <typename U1, typename U2>
    Lattice<T>& operator=(const LatticeExpr<U1, U2>& expr)
    {
      pyQCDassert ((this->data_.size() == expr.size()),
                   std::out_of_range("Array::data_"));
      T* ptr = &(this->data_)[0];
      for (unsigned long i = 0; i < expr.size(); ++i) {
        ptr[i] = static_cast<T>(expr[i]);
      }
      layout_ = &expr.layout();
      site_size_ = expr.site_size();
      return *this;
    }

    Lattice<T>& operator=(const T& rhs)
    {
      data_.assign(data_.size(), rhs);
      return *this;
    }

#define LATTICE_OPERATOR_ASSIGN_DECL(op)				                             \
    template <typename U,                                                    \
      typename std::enable_if<                                               \
		    not std::is_base_of<LatticeObj, U>::value>::type* = nullptr>         \
    Lattice<T>& operator op ## =(const U& rhs);	                             \
    template <typename U>                                                    \
    Lattice<T>& operator op ## =(const Lattice<U>& rhs);

    LATTICE_OPERATOR_ASSIGN_DECL(+);
    LATTICE_OPERATOR_ASSIGN_DECL(-);
    LATTICE_OPERATOR_ASSIGN_DECL(*);
    LATTICE_OPERATOR_ASSIGN_DECL(/);

    unsigned long size() const { return data_.size(); }
    unsigned int volume() const { return layout_->volume(); }
    unsigned int num_dims() const { return layout_->num_dims(); }
    const Site& shape() const
    { return layout_->shape(); }
    const Layout& layout() const { return *layout_; }
    Int site_size() const { return site_size_; }

  protected:
    Int site_size_;
    const Layout* layout_;
    aligned_vector<T> data_;
  };


  template <typename T>
  Lattice<T>& Lattice<T>::operator=(
    const Lattice<T>& lattice)
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
  template <typename T>                                                     \
  template <typename U,                                                     \
    typename std::enable_if<                                                \
      not std::is_base_of<LatticeObj, U>::value>::type*>                    \
  Lattice<T>& Lattice<T>::operator op ## =(const U& rhs)                    \
  {                                                                         \
    for (auto& item : data_) {                                              \
      item op ## = rhs;                                                     \
    }                                                                       \
    return *this;                                                           \
  }                                                                         \
                                                                            \
                                                                            \
  template <typename T>                                                     \
  template <typename U>                                                     \
  Lattice<T>&                                                               \
  Lattice<T>::operator op ## =(const Lattice<U>& rhs)                       \
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
