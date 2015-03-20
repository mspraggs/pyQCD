#define CATCH_CONFIG_MAIN

#include <base/lattice.hpp>

#include "helpers.hpp"

typedef pyQCD::Lattice<double> Lattice;

class TestLayout : public pyQCD::Layout
{
public:
  TestLayout(const std::vector<unsigned int>& shape)
    : pyQCD::Layout(shape, [] (const unsigned int i) { return i; })
  { }
};

TEST_CASE("Lattice test") {
  pyQCD::LexicoLayout layout(std::vector<unsigned int>{8, 4, 4, 4});
  TestLayout another_layout(std::vector<unsigned int>{8, 4, 4, 4});

  Lattice lattice1(&layout, 1.0);
  Lattice lattice2(&layout, 2.0);
  Lattice bad_lattice(&another_layout, 2.0);

  SECTION("Test arithmetic operators") {
    Lattice lattice3 = lattice1 + lattice2;
    for (auto val : lattice3) {
      REQUIRE(val == 3.0);
    }
    REQUIRE(lattice3.layout() == lattice1.layout());
    REQUIRE_THROWS(lattice1 + bad_lattice);
  }

  SECTION("Test accessors") {
    lattice1[0] = 500.0;
    REQUIRE(lattice1(0) == 500.0);
    REQUIRE(lattice1(std::vector<unsigned int>{0, 0, 0, 0}) == 500.0);
  }
}