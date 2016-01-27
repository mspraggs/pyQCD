#define CATCH_CONFIG_MAIN

#include <core/layout.hpp>

#include "helpers.hpp"



TEST_CASE("LexicoLayout test") {
  typedef pyQCD::LexicoLayout Layout;

  Layout layout({8, 4, 4, 4});
  auto subset = layout.subset<Layout>(
    [] (const pyQCD::Int i)
    {
      return (i % 2) ? false : true;
    });

  for (int i = 0; i < 512; ++i) {
    REQUIRE (layout.get_array_index(i) == i);
    REQUIRE (layout.get_site_index(i) == i);
  }
  REQUIRE (layout.get_array_index(pyQCD::Site{4, 3, 2, 1})
             == 313);
  REQUIRE (layout.volume() == 512);
  REQUIRE (layout.num_dims() == 4);
  REQUIRE ((layout.shape() == pyQCD::Site{8, 4, 4, 4}));

  for (int i = 0; i < 256; ++i) {
    REQUIRE (subset.get_site_index(i) == 2 * i);
  }
}