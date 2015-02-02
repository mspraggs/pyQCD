#define CATCH_CONFIG_MAIN

#include <base/array.hpp>

#include "helpers.hpp"

typedef pyQCD::Array<double> Arr;

TEST_CASE("Array test") {
  Arr array1(100, 1.0);
  Arr array2(100, 2.0);

  SECTION ("Testing array constructors") {
    REQUIRE (array1.size() == 100);
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array1[i] == 1.0);
    }
  }

  SECTION("Testing array iterators") {
    for (auto elem : array1) {
      REQUIRE (elem == 1.0);
    }
  }

  SECTION ("Testing array-array binary ops") {
    Arr array3 = array1 + array2;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array3[i] == 3.0);
    }
    array3 = array1 - array2;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array3[i] == -1.0);
    }
    array3 = array1 * array2;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array3[i] == 2.0);
    }
    array3 = array1 / array2;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array3[i] == 0.5);
    }
  }

  SECTION ("Testing array-scalar binary ops") {
    Arr array3 = array1 * 5.0;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array3[i] == 5.0);
    }
    array3 = 5.0 * array1;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array3[i] == 5.0);
    }
    array3 = array1 / 5.0;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array3[i] == 0.2);
    }
  }
}