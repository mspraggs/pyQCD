#define CATCH_CONFIG_MAIN

#include <Eigen/Dense>

#include <core/lattice.hpp>
#include <core/detail/operators.hpp>

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
  std::vector<std::function<double(const double&, const double&)> >
    scalar_operators{
    Plus::apply<double, double>, Minus::apply<double, double>,
    Multiplies::apply<double, double>, Divides::apply<double, double>
  };

  pyQCD::LexicoLayout layout({8, 4, 4, 4});
  TestLayout another_layout({8, 4, 4, 4});

  Lattice lattice1(layout, 1.0, 4);
  Lattice lattice2(layout, 2.0, 4);
  Lattice bad_lattice(another_layout);
  for (unsigned int i = 0; i < bad_lattice.size(); ++i) {
    bad_lattice[i] = i;
  }

  pyQCD::Lattice<Eigen::Matrix3cd> lattice_matrix(
    layout, Eigen::Matrix3cd::Identity() * 4.0);

  SECTION ("Testing array iterators") {
    for (auto elem : lattice1) {
      REQUIRE (elem == 1.0);
    }
  }

  SECTION ("Testing scalar assign") {
    lattice1 = 4.0;
    for (unsigned int i = 0; i < lattice1.size(); ++i) {
      REQUIRE (lattice1[i] == 4.0);
    }
  }

  SECTION ("Testing array-array binary ops") {
    std::vector<std::function<Lattice(const Lattice&, const Lattice&)> >
      lattice_operators{
      Plus::apply<Lattice, Lattice>, Minus::apply<Lattice, Lattice>,
      Multiplies::apply<Lattice, Lattice>, Divides::apply<Lattice, Lattice>
    };
    for (unsigned int i = 0; i < 4; ++i) {
      Lattice array3 = lattice_operators[i](lattice1, lattice2);
      for (unsigned int j = 0; j < lattice1.size(); ++j) {
        REQUIRE (array3[j] == scalar_operators[i](1.0, 2.0));
      }
    }
  }

  SECTION ("Testing array-scalar binary ops") {
    std::vector<std::function<Lattice(const Lattice&, const double&)> >
      lattice_operators{
      Plus::apply<Lattice, double>, Minus::apply<Lattice, double>,
      Multiplies::apply<Lattice, double>, Divides::apply<Lattice, double>
    };
    for (unsigned int i = 0; i < 4; ++i) {
      Lattice lattice3 = lattice_operators[i](lattice1, 2.0);
      for (unsigned int j = 0; j < lattice1.size(); ++j) {
        REQUIRE (lattice3[j] == scalar_operators[i](1.0, 2.0));
      }
    }
  }

  SECTION ("Testing array-array op-assigns") {
    std::vector<std::function<void(Lattice&, const Lattice&)> >
      lattice_operators{
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 += lattice2; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 -= lattice2; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 *= lattice2; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 /= lattice2; }
    };
    for (unsigned int i = 0; i < 4; ++i) {
      auto lattice3 = lattice1;
      lattice_operators[i](lattice3, lattice2);
      for (unsigned int j = 0; j < lattice1.size(); ++j) {
        REQUIRE(lattice3[j] == scalar_operators[i](1.0, 2.0));
      }
    }
  }

  SECTION ("Testing array-scalar op-assigns") {
    std::vector<std::function<void(Lattice&, const Lattice&)> >
      lattice_operators{
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 += 2.0; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 -= 2.0; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 *= 2.0; },
      [] (Lattice& lattice1, const Lattice& lattice2) { lattice1 /= 2.0; }
    };
    for (unsigned int i = 0; i < 4; ++i) {
      auto lattice3 = lattice1;
      lattice_operators[i](lattice3, lattice2);
      for (unsigned int j = 0; j < lattice1.size(); ++j) {
        REQUIRE(lattice3[j] == scalar_operators[i](1.0, 2.0));
      }
    }
  }

  SECTION("Test accessors") {
    lattice1[0] = 500.0;
    REQUIRE(lattice1(0) == 500.0);
    REQUIRE(lattice1(pyQCD::Site{0, 0, 0, 0}) == 500.0);
    lattice1(pyQCD::Site{4, 2, 3, 1}) = 123.0;
    REQUIRE(lattice1(301) == 123.0);
    REQUIRE(lattice1[1204] == 123.0);
  }

  SECTION("Test properties") {
    REQUIRE(&lattice1.layout() == &layout);
    REQUIRE(lattice1.size() == 2048);
    REQUIRE(lattice1.num_dims() == 4);
  }

  SECTION("Test non-scalar site types") {
    MatrixCompare<Eigen::Matrix3cd> comparison(1e-5, 1e-8);
    decltype(lattice_matrix) result
      = lattice_matrix * (3.0 * Eigen::Matrix3cd::Identity());
    REQUIRE(result.size() == lattice_matrix.size());
    for (auto& site_matrix : result) {
      comparison(site_matrix, Eigen::Matrix3cd::Identity() * 12.0);
    }
  }

  SECTION("Test lattice views") {
    auto site_view1 = lattice1.site_view(pyQCD::Site{0, 0, 0, 0});
    auto site_view2 = lattice1.site_view(pyQCD::Site{0, 0, 0, 1});
    site_view1[0] = 5.0;
    REQUIRE(lattice1[0] == 5.0);
    site_view1 = site_view2;// + site_view2;
    REQUIRE(lattice1[0] == 1.0);
  }
}

TEST_CASE("Non-integral Array types test") {
  pyQCD::LexicoLayout layout(std::vector<unsigned int>{8, 4, 4, 4});
  pyQCD::Lattice<Eigen::Matrix3cd> array1(layout, Eigen::Matrix3cd::Identity());
  Eigen::Vector3cd vec(1.0, 1.0, 1.0);
  pyQCD::Lattice<Eigen::Vector3cd> vecs = array1 * vec;
}
