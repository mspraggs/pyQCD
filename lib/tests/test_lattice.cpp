#define CATCH_CONFIG_MAIN

#include <base/lattice.hpp>

#include "helpers.hpp"

typedef pyQCD::Lattice<double> Lattice;

class TestLayout : public pyQCD::Layout
{
public:
  TestLayout(const std::vector<unsigned int>& shape)
    : pyQCD::Layout(shape, [&] (const unsigned int i)
  {
    unsigned int volume = std::accumulate(shape.begin(), shape.end(), 1u,
      std::multiplies<unsigned int>());
    return volume - i - 1; })
  { }
};

TEST_CASE("Lattice test") {
  pyQCD::LexicoLayout layout(std::vector<unsigned int>{8, 4, 4, 4});
  TestLayout another_layout(std::vector<unsigned int>{8, 4, 4, 4});

  Lattice lattice1(&layout, 1.0);
  Lattice lattice2(&layout, 2.0);
  Lattice bad_lattice(&another_layout);
  for (unsigned int i = 0; i < bad_lattice.volume(); ++i) {
    bad_lattice[i] = i;
  }

  SECTION("Test arithmetic operators") {
    Lattice lattice3 = lattice1 + lattice2;
    for (auto val : lattice3) {
      REQUIRE(val == 3.0);
    }
    REQUIRE(lattice3.layout() == lattice1.layout());
    REQUIRE_THROWS(lattice1 + bad_lattice);
  }

  SECTION("Test assignment operator") {
    lattice1 = bad_lattice;
    REQUIRE(lattice1[0] == bad_lattice[511]);
    std::vector<unsigned int> site{0, 0, 0, 0};
    REQUIRE(lattice1(site) == bad_lattice(site));
  }

  SECTION("Test accessors") {
    lattice1[0] = 500.0;
    REQUIRE(lattice1(0) == 500.0);
    REQUIRE(lattice1(std::vector<unsigned int>{0, 0, 0, 0}) == 500.0);
    lattice1(std::vector<unsigned int>{4, 2, 3, 1}) = 123.0;
    REQUIRE(lattice1(301) == 123.0);
    REQUIRE(lattice1[301] == 123.0);
  }

  SECTION("Test properties") {
    REQUIRE(lattice1.layout() == &layout);
    REQUIRE(lattice1.volume() == 512);
    REQUIRE(lattice1.num_dims() == 4);
  }
}