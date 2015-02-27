#define CATCH_CONFIG_MAIN

#include <base/lattice.hpp>

#include "helpers.hpp"

typedef pyQCD::Lattice<double> Lattice;

TEST_CASE("Lattice test") {
  pyQCD::LexicoLayout(std::vector<unsigned int>{8, 4, 4, 4});
}