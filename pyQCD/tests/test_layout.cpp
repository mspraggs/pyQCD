#define CATCH_CONFIG_MAIN

#include <core/layout.hpp>

#include "helpers.hpp"



TEST_CASE("LexicoLayout test") {
  typedef pyQCD::LexicoLayout Layout;

  Layout layout({8, 4, 4, 4});
  auto subset = layout.subset<Layout>(
    [] (const Layout::Int i)
    {
      return (i % 2) ? false : true;
    });

  for (int i = 0; i < 512; ++i) {
    REQUIRE (layout.get_array_index(i) == i);
    REQUIRE (layout.get_site_index(i) == i);
  }
  REQUIRE (layout.get_array_index(std::vector<Layout::Int>{4, 3, 2, 1})
             == 313);
  REQUIRE (layout.volume() == 512);
  REQUIRE (layout.num_dims() == 4);
  REQUIRE ((layout.shape() == std::vector<unsigned int>{8, 4, 4, 4}));

  for (int i = 0; i < 256; ++i) {
    REQUIRE (subset.get_site_index(i) == 2 * i);
  }
}