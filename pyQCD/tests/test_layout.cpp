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
 *
 * Tests for the Layout class.
 */

#include <core/layout.hpp>

#include "helpers.hpp"



TEST_CASE("LexicoLayout test") {
  using Layout = pyQCD::LexicoLayout;

  Layout layout({8, 4, 4, 4});

  for (int i = 0; i < 512; ++i) {
    REQUIRE (layout.get_array_index(i) == i);
    REQUIRE (layout.get_site_index(i) == i);
  }
  REQUIRE (layout.get_array_index(pyQCD::Site{4, 3, 2, 1}) == 313);
  REQUIRE (layout.volume() == 512);
  REQUIRE (layout.num_dims() == 4);
  REQUIRE ((layout.shape() == pyQCD::Site{8, 4, 4, 4}));
  REQUIRE (layout.is_even_site(0));
  REQUIRE (layout.is_even_site(pyQCD::Site{2, 0, 1, 1}));
}

TEST_CASE("EvenOddLayout test") {
  using Layout = pyQCD::EvenOddLayout;

  Layout layout({8, 4, 4, 4});

  REQUIRE (layout.get_array_index(0) == 0);
  REQUIRE (layout.get_site_index(0) == 0);

  REQUIRE (layout.get_array_index(1) == 256);
  REQUIRE (layout.get_site_index(256) == 1);

  REQUIRE (layout.get_array_index(2) == 1);
  REQUIRE (layout.get_site_index(1) == 2);

  REQUIRE (layout.get_array_index(3) == 257);
  REQUIRE (layout.get_site_index(257) == 3);

  REQUIRE (layout.get_array_index(4) == 258);
  REQUIRE (layout.get_site_index(258) == 4);
}


TEST_CASE("PartitionCompare test") {
  using Layout = pyQCD::LexicoLayout;

  Layout layout({8, 4, 4, 4});

  pyQCD::PartitionCompare compare(2, layout);

  REQUIRE(not compare(0, 0));

  for (unsigned int i = 1; i < 10; ++i) {
    REQUIRE(compare(0, i));
  }

  REQUIRE(not compare(0, 10));

  for (unsigned int i = 11; i < 34; ++i) {
    REQUIRE(compare(0, i));
  }

  REQUIRE(not compare(0, 34));

  REQUIRE(compare(7, 510));
}