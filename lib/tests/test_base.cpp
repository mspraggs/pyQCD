#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_base
#include <vector>
#include <exception>
#include <random>
#include <functional>

#include <boost/test/unit_test.hpp>
#include <boost/test/parameterized_test.hpp>
#include <boost/bind.hpp>

#include <base/lattice_base.hpp>

typedef pyQCD::LatticeBase<double> BaseDouble;

struct TestRandom
{
  TestRandom()
  {
    BOOST_TEST_MESSAGE("Set up random number generator");
    gen = std::mt19937(rd());
    real_dist = std::uniform_real_distribution<>(0, 10);
    int_dist = std::uniform_int_distribution<>(0, 100);
  }
  ~TestRandom() { BOOST_TEST_MESSAGE("Tear down random number generator"); }
  int gen_int() { return int_dist(gen); }
  double gen_real() { return real_dist(gen); }

  std::random_device rd;
  std::mt19937 gen;
  std::uniform_real_distribution<> real_dist;
  std::uniform_int_distribution<> int_dist;
};

struct TestLayout
{
  TestLayout()
  {
    BOOST_TEST_MESSAGE("Set up test layout");
    lattice_volume = 1;
    block_volume = 1;
    for (int i = 0; i < NDIM; ++i) {
      if (i == 0)
	lattice_shape.push_back(8);
      else
	lattice_shape.push_back(4);
      bad_latt_shape.push_back(6);
      block_shape.push_back(4);
      lattice_volume *= lattice_shape[i];
      block_volume *= block_shape[i];
      num_blocks = lattice_volume / block_volume;
    }
  }
  ~TestLayout() { BOOST_TEST_MESSAGE("Tear down test layout"); }

  std::vector<int> lattice_shape;
  std::vector<int> bad_latt_shape;
  std::vector<int> block_shape;
  int lattice_volume;
  int block_volume;
  int num_blocks;
};



void constructor_test(const BaseDouble& lattice_base,
		      const TestLayout& layout)
{
  BOOST_REQUIRE_EQUAL_COLLECTIONS(lattice_base.lattice_shape().begin(),
				  lattice_base.lattice_shape().end(),
				  layout.lattice_shape.begin(),
				  layout.lattice_shape.end());
  BOOST_REQUIRE_EQUAL_COLLECTIONS(lattice_base.block_shape().begin(),
				  lattice_base.block_shape().end(),
				  layout.block_shape.begin(),
				  layout.block_shape.end());
  BOOST_REQUIRE_EQUAL(lattice_base.layout().size(), layout.lattice_volume);
  BOOST_REQUIRE_EQUAL(lattice_base.lattice_volume(), layout.lattice_volume);
  BOOST_REQUIRE_EQUAL(lattice_base.block_volume(), layout.block_volume);
  BOOST_REQUIRE_EQUAL(lattice_base.num_blocks(), layout.num_blocks);
}



void const_value_test(const BaseDouble& lattice_base,
		      const double value)
{
  for (int i = 0; i < lattice_base.num_blocks(); ++i)
    for (int j = 0; j < lattice_base.block_volume(); ++j)
      BOOST_CHECK_CLOSE(lattice_base.datum_ref(i, j), value, 1e-8);
}



BOOST_AUTO_TEST_SUITE(test_lattice_base)

BOOST_AUTO_TEST_CASE(test_lattice_base_constructors)
{
  TestLayout layout;
  TestRandom rng;
  double rand_num = rng.gen_real();
  BaseDouble test_const_init(rand_num, layout.lattice_shape,
			     layout.block_shape);

  std::vector<BaseDouble> test_bases;
  test_bases.push_back(BaseDouble(layout.lattice_shape, layout.block_shape));
  test_bases.push_back(test_const_init);
  test_bases.push_back(BaseDouble(test_const_init));
  BaseDouble test_equals = test_const_init;
  test_bases.push_back(test_equals);

  boost::unit_test::callback1<BaseDouble> bound_constructor_test
    = boost::bind(&constructor_test, _1, layout);
  BOOST_PARAM_TEST_CASE(bound_constructor_test,
			test_bases.begin(), test_bases.end());

  BOOST_REQUIRE_THROW(BaseDouble bad_base(layout.bad_latt_shape,
					  layout.block_shape),
		    std::invalid_argument);

  boost::unit_test::callback1<BaseDouble> bound_const_value_test
    = boost::bind(&const_value_test, _1, rand_num);
  BOOST_PARAM_TEST_CASE(bound_const_value_test,
			test_bases.begin() + 1, test_bases.end());
}

BOOST_AUTO_TEST_CASE(test_lattice_base_utils)
{
  TestLayout layout;
  BaseDouble test_base(layout.lattice_shape, layout.block_shape);

  for (int i = 0; i < layout.lattice_volume; ++i) {
    std::vector<int> coords = test_base.get_site_coords(i);
    for (int j = 0; j < NDIM; ++j) {
      BOOST_REQUIRE(coords[j] >= 0);
      BOOST_REQUIRE(coords[j] < layout.lattice_shape[j]);
    }
    BOOST_REQUIRE_EQUAL(test_base.get_site_index(coords), i);
    test_base.get_site_coords(i, coords);
    for (int j = 0; j < NDIM; ++j) {
      BOOST_REQUIRE(coords[j] >= 0);
      BOOST_REQUIRE(coords[j] < layout.lattice_shape[j]);
    }
    BOOST_REQUIRE_EQUAL(test_base.get_site_index(coords), i);
  }
}

BOOST_AUTO_TEST_CASE(test_accessors)
{
  TestLayout layout;
  TestRandom rng;
  
  BaseDouble test_base(layout.lattice_shape, layout.block_shape);

  std::vector<double> random_values(layout.lattice_volume, 0.0);
  for (int i = 0; i < layout.lattice_volume; ++i) {
    random_values[i] = rng.gen_real();
    test_base[i] = random_values[i];
    BOOST_CHECK_CLOSE(test_base[i], random_values[i], 1e-8);
  }
  for (int i = 0; i < layout.lattice_volume; ++i) {
    BOOST_CHECK_CLOSE(test_base[i], random_values[i], 1e-8);
  }

  for (int i = 0; i < layout.lattice_volume; ++i) {
    std::vector<int> coords = test_base.get_site_coords(i);
    int n0 = coords[0];
#if NDIM>1
    int n1 = coords[1];
#endif
#if NDIM>2
    int n2 = coords[2];
#endif
#if NDIM>3
    int n3 = coords[3];
#endif
#if NDIM>4
    int n4 = coords[4];
#endif
#if NDIM>5
    int n5 = coords[5];
#endif
#if NDIM>6
    int n6 = coords[6];
#endif
#if NDIM>7
    int n7 = coords[7];
#endif
#if NDIM>8
    int n8 = coords[8];
#endif
#if NDIM>9
    int n9 = coords[9];
#endif
    random_values[i] = rng.gen_real();
    test_base(COORD_INDEX_PARAMS(n)) = random_values[i];
  }
  for (int i = 0; i < layout.lattice_volume; ++i) {
    std::vector<int> coords = test_base.get_site_coords(i);
    int n0 = coords[0];
#if NDIM>1
    int n1 = coords[1];
#endif
#if NDIM>2
    int n2 = coords[2];
#endif
#if NDIM>3
    int n3 = coords[3];
#endif
#if NDIM>4
    int n4 = coords[4];
#endif
#if NDIM>5
    int n5 = coords[5];
#endif
#if NDIM>6
    int n6 = coords[6];
#endif
#if NDIM>7
    int n7 = coords[7];
#endif
#if NDIM>8
    int n8 = coords[8];
#endif
#if NDIM>9
    int n9 = coords[9];
#endif
    BOOST_CHECK_CLOSE(test_base(COORD_INDEX_PARAMS(n)), random_values[i], 1e-8);
  }
}

BOOST_AUTO_TEST_SUITE_END()
