#ifndef PYQCD_HELPERS_HPP
#define PYQCD_HELPERS_HPP

/*
 * This file is part of pyQCD.
 *
 * pyQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * pyQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 *
 * Created by Matt Spraggs on 10/02/16.
 *
 * Utilities to facilitate benchmarking */

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

  if (num_flops > 0) {
    std::cout << "Performed " << num_flops * num_trials << " flops in "
      << elapsed << " seconds";
    std::cout << " => " << num_trials * num_flops / elapsed / 1000000.0
      << " Mflops." << std::endl;
  }
  else {
    std::cout << "Performed " << num_trials << " runs in " << elapsed;
    std::cout << " seconds (" << elapsed / num_trials << " per run)";
    std::cout << std::endl;
  }
}

#endif
