#define CATCH_CONFIG_MAIN

#include <base/lattice.hpp>

#include "helpers.hpp"

typedef pyQCD::Lattice<double> Lattice;

TEST_CASE("Lattice test") {
  pyQCD::LexicoLayout layout(std::vector<unsigned int>{8, 4, 4, 4});

  Lattice lattice1(&layout, 1.0);
  Lattice lattice2(&layout, 2.0);

  SECTION("Test arithmetic operators") {
    Lattice lattice3 = lattice1 + lattice2;
    for (auto val : lattice3) {
      REQUIRE(val == 3.0);
    }
  }
}