#ifndef GAUGE_FIELD_HPP
#define GAUGE_FIELD_HPP

/* This file contains the implementation for GaugeField, which is basically
 * an instance of Array with the element type being an NC x NC Eigen matrix.
 * This object represents the field that resides on links between lattice sites.
 */

#include <Eigen/Dense>

#include "array.hpp"

namespace pyQCD {
  template<int NC, typename T = double>
  class GaugeField
    : public Array<Eigen::Matrix<std::complex<T>, NC, NC>,
      Eigen::aligned_allocator>
  {
  public:
    typedef Eigen::Matrix<std::complex<T>, NC, NC> colour_matrix_type;
    using Array<colour_matrix_type, Eigen::aligned_allocator>::Array;

    const ArrayUnary<Array<colour_matrix_type, Eigen::aligned_allocator>,
      colour_matrix_type, Adjoint>
    adjoint() const
    {
      return ArrayUnary<Array<colour_matrix_type, Eigen::aligned_allocator>,
        colour_matrix_type, Adjoint>(*this);
    }
  };
}
#endif