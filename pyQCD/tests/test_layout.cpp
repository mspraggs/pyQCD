#define CATCH_CONFIG_MAIN

#include <core/layout.hpp>

#include "helpers.hpp"


TEST_CASE("LexicoLayout test") {
  pyQCD::LexicoLayout layout(std::vector<unsigned int>{8, 4, 4, 4});

  for (int i = 0; i < 512; ++i) {
    REQUIRE (layout.get_array_index(i) == i);
    REQUIRE (layout.get_site_index(i) == i);
  }
  REQUIRE (layout.get_array_index(std::vector<unsigned int>{4, 3, 2, 1})
             == 313);
  REQUIRE (layout.volume() == 512);
  REQUIRE (layout.num_dims() == 4);
  REQUIRE ((layout.lattice_shape() == std::vector<unsigned int>{8, 4, 4, 4}));
}
