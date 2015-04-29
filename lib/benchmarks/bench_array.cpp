/* Benchmark for Array type. */

#include <Eigen/Dense>

#include "helpers.hpp"

#include <base/array.hpp>


int main()
{
  int n = 1000000;
  typedef Eigen::Matrix2d arr_type;
  arr_type elem = arr_type::Ones();
  pyQCD::Array<arr_type, Eigen::aligned_allocator> array1(n, elem);
  pyQCD::Array<arr_type, Eigen::aligned_allocator> array2(n, elem);
  pyQCD::Array<arr_type, Eigen::aligned_allocator> array3(n, elem);

  benchmark([&] () {
    decltype(array1) result = array1 + array2 + array3;
  }, 16 * n);

  return 0;
}