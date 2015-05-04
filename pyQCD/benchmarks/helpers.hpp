#ifndef HELPERS_HPP
#define HELPERS_HPP

/* Utilities to facilitate benchmarking */

#include <iostream>
#include <chrono>

unsigned int matmul_flops(
  const unsigned int n, const bool complex, const unsigned int float_width)
{
  const unsigned int mul_flops = complex ? 6 : 1;
  const unsigned int add_flops = complex ? 2 : 1;
  return (n * mul_flops + (n - 1) * add_flops) * n * n * float_width;
}


unsigned int matadd_flops(
  const unsigned int n, const bool complex, const unsigned int float_width)
{
  return n * n * float_width * (complex ? 2 : 1);
}


template <typename Fn>
void benchmark(Fn func, const long num_flops = 0, const int num_trials = 100)
{
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < num_trials; ++i) {
    func();
  }
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  double elapsed = elapsed_seconds.count();
  std::cout << "Performed " << num_flops * num_trials << " flops in " << elapsed
    << " seconds";

  if (num_flops > 0) {
    std::cout << " => " << num_trials * num_flops / elapsed / 1000000.0
      << " Mflops." << std::endl;
  }
  else {
    std::cout << "." << std::endl;
  }
}

#endif
