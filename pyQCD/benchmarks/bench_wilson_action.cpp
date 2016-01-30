/* Benchmark for Gauge::WilsonAction */

#include <gauge/wilson_action.hpp>

#include "helpers.hpp"


int main()
{
  typedef double Real;
  typedef pyQCD::ColourMatrix<Real, 3> ColourMatrix;

  auto identity = ColourMatrix::Identity();
  pyQCD::LexicoLayout layout({8, 4, 4, 4, 4});
  pyQCD::LatticeColourMatrix<double, 3> gauge_field(layout, identity);
  pyQCD::Gauge::WilsonAction<double, 3> action(5.0, layout);

  std::cout << "Benchmarking WilsonAction::compute_staples..." << std::endl;
  benchmark([&] () {
    ColourMatrix staple = action.compute_staples(gauge_field, 0);
  }, 0, 1000000);
}