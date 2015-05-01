/* Benchmark for Array type. */

#include <Eigen/Dense>

#include "helpers.hpp"

#include <base/array.hpp>


template <typename T, template <typename> class Alloc = std::allocator>
void profile_for_type(const T& elem, const std::string& type,
  const int add_flops, const int multiply_flops)
{
  std::cout << "Profiling for array type " << type << "." << std::endl;
  int n = 1000;
  pyQCD::Array<T, Alloc> array1(n, elem);
  decltype(array1) array2(n, elem);
  decltype(array1) array3(n, elem);
  decltype(array1) result(n, elem);

  std::cout << "Profiling f(x, y, z) = x + y + z:" << std::endl;
  benchmark([&] () {
    result = array1 + array2 + array3;
  }, 2 * add_flops * n, 100000);

  std::cout << "Profiling f(x, y) = 5.0 * x + y:" << std::endl;
  benchmark([&] () {
    result = 5.0 * array1 + array2;
  }, 2 * add_flops * n, 100000);

  std::cout << "Profiling f(x, y, z) = x * y + z:" << std::endl;
  benchmark([&] () {
    result = array1 * array2 + array3;
  }, (add_flops + multiply_flops) * n, 100000);
}


int main(int argc, char* argv[])
{
  profile_for_type(1.0, "double", 2, 2);
  profile_for_type<Eigen::Matrix2d, Eigen::aligned_allocator>(
    Eigen::Matrix2d::Random(), "Eigen::Matrix2d", 8, 24
  );
  return 0;
}
