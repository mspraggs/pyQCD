/* Benchmark for Array type. */

#include <Eigen/Dense>

#include "helpers.hpp"

#include <core/lattice.hpp>
#include <core/layout.hpp>


template <typename T, template <typename> class Alloc = std::allocator>
void profile_for_type(const T& elem, const std::string& type,
  const int add_flops, const int multiply_flops)
{
  std::cout << "Profiling for array type " << type << "." << std::endl;

  int n = 100;
  pyQCD::LexicoLayout layout(std::vector<unsigned int>{n});
  pyQCD::Lattice<T, Alloc> array1(layout, elem);
  decltype(array1) array2(layout, elem);
  decltype(array1) array3(layout, elem);
  decltype(array1) result(layout, elem);

  std::cout << "Profiling f(x, y, z) = x + y + z:" << std::endl;
  benchmark([&] () {
    result = array1 + array2 + array3;
  }, 2 * add_flops * n, 1000000);

  std::cout << "Profiling f(x, y) = 5.0 * x + y:" << std::endl;
  benchmark([&] () {
    result = 5.0 * array1 + array2;
  }, 2 * add_flops * n, 1000000);

  std::cout << "Profiling f(x, y, z) = x * y + z:" << std::endl;
  benchmark([&] () {
    result = array1 * array2 + array3;
  }, (add_flops + multiply_flops) * n, 1000000);
  
  std::cout << std::endl;
}


int main(int argc, char* argv[])
{
  profile_for_type(1.0, "double", 2, 2);
  profile_for_type(std::complex<double>(1.0, 0.0), "std::complex<double>",
                   4, 12);
  profile_for_type<Eigen::Matrix2d, Eigen::aligned_allocator>(
    Eigen::Matrix2d::Random(), "Eigen::Matrix2d",
    matadd_flops(2, false, 2), matmul_flops(2, false, 2)
  );
  profile_for_type<Eigen::Matrix4d, Eigen::aligned_allocator>(
    Eigen::Matrix4d::Random(), "Eigen::Matrix4d",
    matadd_flops(4, false, 2), matmul_flops(4, false, 2)
  );
  profile_for_type<Eigen::Matrix2cd, Eigen::aligned_allocator>(
    Eigen::Matrix2cd::Random(), "Eigen::Matrix2cd",
    matadd_flops(2, true, 2), matmul_flops(2, true, 2)
  );
  profile_for_type<Eigen::Matrix3cd, Eigen::aligned_allocator>(
    Eigen::Matrix3cd::Random(), "Eigen::Matrix3cd",
    matadd_flops(3, true, 2), matmul_flops(3, true, 2)
  );
  return 0;
}
