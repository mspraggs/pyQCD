/* Benchmark for Array type. */

#include <Eigen/Dense>

#include "helpers.hpp"

#include <base/array.hpp>


int main(int argc, char* argv[])
{
  int n = 1000;
  typedef Eigen::Matrix2d arr_type;
  arr_type elem = arr_type::Ones();
  pyQCD::Array<arr_type, Eigen::aligned_allocator> array1(n, elem);
  pyQCD::Array<arr_type, Eigen::aligned_allocator> array2(n, elem);
  pyQCD::Array<arr_type, Eigen::aligned_allocator> array3(n, elem);

  std::cout << "Profiling f(x, y, z) = x + y + z:" << std::endl;
  benchmark([&] () {
    decltype(array1) result = array1 + array2 + array3;
  }, 16 * n, 100000);

  std::cout << "Profiling f(x, y) = a * x + y:" << std::endl;
  benchmark([&] () {
    decltype(array1) result = 5.0 * array1 + array2;
  }, 16 * n, 100000);

  std::cout << "Profiling f(x, y, z) = x * y + z:" << std::endl;
  benchmark([&] () {
    decltype(array1) result = array1 * array2 + array3;
  }, 32 * n, 100000);

  return 0;
}