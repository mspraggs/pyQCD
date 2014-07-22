#ifndef SITE_GAUGE_FIELD_HPP
#define SITE_GAUGE_FIELD_HPP

/* This file provides a container for the gauge link objects attached to a
 * particular lattice site.
 *
 * The SiteGaugeField is actually a template with template parameters
 * corresponding to the number of rows and columns in the gauge link type, as
 * well as the number of dimensions.
 *
 * The expression templates to optimize the arithmetic for this class can be
 * found in site_gauge_field_expr.hpp.
 */

#include <vector>

#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace pyQCD
{
  template <int nrows, int ncols, int ndim>
  class SiteGaugeField
  {
  public:
    // Typedefs
    typedef Eigen::Matrix<std::complex<double>, nrows, ncols> link_type;
    // Constructors
    SiteGaugeField();
    SiteGaugeField(const link_type& link_val);
    SiteGaugeField(const SiteGaugeField<nrows, ncols, ndim>& site_gauge_field);
    ~SiteGaugeField();

  protected:
    // The vector that holds the link variables
    std::vector<link_type, Eigen::aligned_allocator<link_type> > _data;
  };



  template <int nrows, int ncols, int ndim>
  SiteGaugeField<nrows, ncols, ndim>::SiteGaugeField()
  {
    // Default constructor - we initialize the link variables as the identity
    // elements if we're dealing with square matrices, and we fill it with ones
    // if it's not.
    this->_data.resize(ndim);
    for (link_type& link : this->_data) {
      if (nrows == ncols)
	link = link_type::Identity();
      else
	link = link_type::Ones();
    }
  }



  template <int nrows, int ncols, int ndim>
  SiteGaugeField<nrows, ncols, ndim>::SiteGaugeField(const link_type& link_val)
  {
    // Constructor - initialize all link values to a constant value
    this->_data.resize(ndim);
    for (link_type& link : this->_data)
	link = link_val;
  }



  template <int nrows, int ncols, int ndim>
  SiteGaugeField<nrows, ncols, ndim>::SiteGaugeField(
    const SiteGaugeField<nrows, ncols, ndim>& site_gauge_field)
    : _data(site_gauge_field._data)
  {
    // Copy constructor. Nothing else to do here.
  }



  template <int nrows, int ncols, int ndim>
  SiteGaugeField<nrows, ncols, ndim>::~SiteGaugeField()
  {
    // Destructor - nothing to do here.
  }
}

#endif
