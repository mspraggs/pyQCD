#ifndef PYQCD_LATTICE_HPP
#define PYQCD_LATTICE_HPP

/* This file declares and defines the Lattice class. This is basically an Array
 * but with a Layout member specifying the relationship between the sites and
 * the Array index. In addition, there are operator() implementations to access
 * elements using site coordinates or a lexicographic index.
 */

#include <cassert>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

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
    Lattice(const Layout& layout, const Int site_size = 1);
    Lattice(const Layout& layout, const T& val, const Int site_size = 1);
    Lattice(const Lattice<T>& lattice);
    template <typename U1, typename U2>
    Lattice(const LatticeExpr<U1, U2>& expr)
    {
      this->data_.resize(expr.size());
      for (unsigned long i = 0; i < expr.size(); ++i) {
        this->data_[i] = static_cast<T>(expr[i]);
      }
      layout_ = &expr.layout();
      site_size_ = expr.site_size();
      init_mpi_types();
      init_mpi_status();
      init_mpi_pointers();
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
    unsigned int local_volume() const { return layout_->local_volume(); }
    unsigned int num_dims() const { return layout_->num_dims(); }
    const Site& local_shape() const
    { return layout_->local_shape(); }
    const Layout& layout() const { return *layout_; }
    Int site_size() const { return site_size_; }

  protected:
    // Constructor helpers
    void init_mpi_types();
    void init_mpi_status();
    void init_mpi_pointers();

    // Destructor helper
    void destruct_mpi_types();

    // Member variables
    Int site_size_;
    bool mpi_types_constructed_;
    const Layout* layout_;
    aligned_vector<T> data_;

    std::vector<T*> comms_buffers_;
    std::vector<T*> surface_pointers_;

    MPI_Datatype site_mpi_type_;
    std::vector<MPI_Datatype> buffer_mpi_types_, surface_mpi_types_;
    std::vector<MPI_Request> send_requests_, recv_requests_;
    std::vector<MPI_Status> send_status_, recv_status_;
  };


  template <typename T>
  Lattice<T>::Lattice(const Layout& layout, const Int site_size)
    : Lattice(layout, T(), site_size)
  { }


  template <typename T>
  Lattice<T>::Lattice(const Layout& layout,  const T& val, const Int site_size)
  : site_size_(site_size), layout_(&layout),
    data_(site_size_ * layout.local_size(), val)
  {
    init_mpi_types();
    init_mpi_status();
    init_mpi_pointers();
  }


  template <typename T>
  Lattice<T>::Lattice(const Lattice<T>& lattice)
    : Lattice(lattice.layout(), T(), lattice.site_size_)
  {
    data_ = lattice.data_;
  }


  template <typename T>
  void Lattice<T>::init_mpi_types()
  {
    if (mpi_types_constructed_) {
      return;
    }
    // Initialise the MPI_Datatype objects that describe the site type, the
    // surface types and the buffer types.
    site_mpi_type_ = MpiType<T>::make(site_size_);
    MPI_Type_commit(&site_mpi_type_);
    // May be able to trim the number of these down in future
    // TODO: Trim these down if possible - there may be duplication
    buffer_mpi_types_.resize(layout_->num_buffers());
    surface_mpi_types_.resize(layout_->num_buffers());

    for (unsigned int buffer = 0; buffer < layout_->num_buffers(); ++buffer) {
      // Construct MPI types for this buffer
      auto surface_site_indices = layout_->surface_site_offsets(buffer);
      std::vector<int> blocklengths(surface_site_indices.size(), 1);
      // Create the types themselves
      MPI_Type_contiguous(layout_->buffer_volume(buffer), site_mpi_type_,
                          &buffer_mpi_types_[buffer]);
      MPI_Type_indexed(static_cast<int>(surface_site_indices.size()),
                       blocklengths.data(),
                       static_cast<int*>(surface_site_indices.data()),
                       site_mpi_type_, &surface_mpi_types_[buffer]);
      // Commit the types
      // TODO: Call MPI_Type_free for these types in destructor
      MPI_Type_commit(&buffer_mpi_types_[buffer]);
      MPI_Type_commit(&surface_mpi_types_[buffer]);
    }
    mpi_types_constructed_ = true;
  }


  template <typename T>
  void Lattice<T>::init_mpi_status()
  {
    // Resize status and request vectors to the number of buffers we have
    send_requests_.resize(layout_->num_buffers());
    recv_requests_.resize(layout_->num_buffers());
    send_status_.resize(layout_->num_buffers());
    recv_status_.resize(layout_->num_buffers());
  }


  template <typename T>
  void Lattice<T>::init_mpi_pointers()
  {
    // Initialise pointers that point to first array element in data_
    comms_buffers_.resize(layout_->num_buffers());
    surface_pointers_.resize(layout_->num_buffers());

    comms_buffers_[0] = data_.data() + layout_->local_volume();
    surface_pointers_[0] = data_.data() + layout_->surface_site_corner_index(0);
    for (unsigned int buffer = 1; buffer < layout_->num_buffers(); ++buffer) {
      comms_buffers_[buffer]
        = comms_buffers_[buffer - 1] + layout_->buffer_volume(buffer - 1);
      surface_pointers_[buffer]
        = data_.data() + layout_->surface_site_corner_index(buffer);
    }
  }


  template <typename T>
  void Lattice<T>::destruct_mpi_types()
  {
    // Run MPI_Type_free over vector of MPI_Datatype
    if (mpi_types_constructed_) {
      MPI_Type_free(&site_mpi_type_);
      for (auto& type : buffer_mpi_types_) {
        MPI_Type_free(&type);
      }
      for (auto& type : surface_mpi_types_) {
        MPI_Type_free(&type);
      }
    }
    mpi_types_constructed_ = false;
  }


  template <typename T>
  Lattice<T>& Lattice<T>::operator=(const Lattice<T>& lattice)
  {
    if (&lattice != this) {
      // TODO: Sort out the logic here - just do a raw assign without checks
      if (layout_) {
        pyQCDassert (lattice.size() == size(),
          std::invalid_argument("lattice.volume() != volume()"));
      }
      else {
        destruct_mpi_types();
        layout_ = lattice.layout_;
        init_mpi_types();
        init_mpi_pointers();
        init_mpi_status();
      }
      for (unsigned int i = 0; i < data_.size(); ++i) {
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
