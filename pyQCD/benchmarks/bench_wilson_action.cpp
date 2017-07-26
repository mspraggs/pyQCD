/*
 * This file is part of pyQCD.
 *
 * pyQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
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
 * Benchmark for gauge::WilsonAction */

#include <gauge/wilson_action.hpp>

#include "helpers.hpp"


int main()
{
  using Real = double;
  using ColourMatrix = pyQCD::ColourMatrix<Real, 3>;

  auto identity = ColourMatrix::Identity();
  pyQCD::LexicoLayout layout({8, 4, 4, 4, 4});
  pyQCD::LatticeColourMatrix<double, 3> gauge_field(layout, identity);
  pyQCD::gauge::WilsonAction<double, 3> action(5.0, layout);

  std::cout << "Benchmarking WilsonAction::compute_staples..." << std::endl;
  benchmark([&] () {
    ColourMatrix staple = action.compute_staples(gauge_field, 0);
  }, 0, 1000000);
}