#ifndef FERMION_HPP
#define FERMION_HPP

/* Here we implement a fermion for a single site, which is basically just an
 * array of Eigen vector objects. */

#include <Eigen/Dense>

#include "array.hpp"


namespace pyQCD
{
  template <int NC, typename T = double>
  class Fermion
    : public Array<Eigen::Matrix<std::complex<T>, NC, 1>,
      Eigen::aligned_allocator>
  {
  public:
    typedef Eigen::Matrix<std::complex<T>, NC, 1> colour_vector_type;
    using Array<colour_vector_type, Eigen::aligned_allocator>::Array;

    const ArrayUnary<Array<colour_vector_type, Eigen::aligned_allocator>,
      colour_vector_type, Adjoint>
    adjoint() const
    {
      return ArrayUnary<Array<colour_vector_type, Eigen::aligned_allocator>,
        colour_vector_type, Adjoint>(*this);
    }
  };
}

#endif