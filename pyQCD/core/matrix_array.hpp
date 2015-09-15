#ifndef FERMION_HPP
#define FERMION_HPP

/* Here we implement an array of matrix objects for a single lattice site site,
 * which is basically just an array of Eigen vector objects.
 */

#include <Eigen/Dense>

#include "array.hpp"


namespace pyQCD
{
  template <int N, int M, typename T = double>
  class MatrixArray
    : public Array<Eigen::Matrix<std::complex<T>, N, M>,
      Eigen::aligned_allocator>
  {
  public:
    typedef Eigen::Matrix<std::complex<T>, N, M> colour_vector_type;
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