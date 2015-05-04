#ifndef HELPERS_HPP
#define HELPERS_HPP

/* Utilities to facilitate benchmarking */

#include <iostream>
#include <chrono>

template <typename Fn>
void benchmark(Fn func, const long num_flops = 0, const int num_trials = 100)
{
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < num_trials; ++i) {
    func();
  }
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  double elapsed = elapsed_seconds.count() / num_trials;
  std::cout << "Performed " << num_flops << " flops in " << elapsed
    << " seconds";

  if (num_flops > 0) {
    std::cout << " => " << num_flops / elapsed / 1000000.0 << " Mflops."
      << std::endl;
  }
  else {
    std::cout << "." << std::endl;
  }
}

#endif
