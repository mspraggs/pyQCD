#define CATCH_CONFIG_MAIN

#include <Eigen/Dense>

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
    array3 = array1 / 5.0 + array2;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array3[i] == 2.2);
    }
  }

  SECTION ("Testing array-array op-assigns") {
    array1 += array2;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array1[i] == 3.0);
    }
    array1 -= array2;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array1[i] == 1.0);
    }
    array1 *= array2;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array1[i] == 2.0);
    }
    array1 /= array2;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array1[i] == 1.0);
    }
  }

  SECTION ("Testing array-scalar op-assigns") {
    array1 += 5.0;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array1[i] == 6.0);
    }
    array1 -= 3.0;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array1[i] == 3.0);
    }
    array1 *= 2.0;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array1[i] == 6.0);
    }
    array1 /= 3.0;
    for (int i = 0; i < 100; ++i) {
      REQUIRE (array1[i] == 2.0);
    }
  }
}

TEST_CASE("Non-integral Array types test") {
  pyQCD::Array<Eigen::Matrix3cd> array1(4, Eigen::Matrix3cd::Identity());
  Eigen::Vector3cd vec(1.0, 1.0, 1.0);
  pyQCD::Array<Eigen::Vector3cd> vecs = array1 * vec;
  pyQCD::Array<pyQCD::Array<Eigen::Matrix3cd> > array2(10, array1);
  array2 *= 3.0;
}