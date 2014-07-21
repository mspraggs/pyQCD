#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_lattice_base
#include <vector>
#include <exception>
#include <functional>

#include <boost/test/unit_test.hpp>
#include <boost/test/parameterized_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>

#include <base/lattice_base.hpp>

#include "random.hpp"

typedef pyQCD::LatticeBase<double> BaseDouble;

boost::test_tools::close_at_tolerance<double>
fp_compare(boost::test_tools::percent_tolerance(1e-8));

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
  bool result = true;
  for (int i = 0; i < lattice_base.num_blocks(); ++i)
    for (int j = 0; j < lattice_base.block_volume(); ++j)
      if (not fp_compare(lattice_base.datum_ref(i, j), value)) {
	result = false;
	break;
      }
  BOOST_CHECK(result);
}



BOOST_AUTO_TEST_SUITE(test_lattice_base)

BOOST_AUTO_TEST_CASE(test_constructors)
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

BOOST_AUTO_TEST_CASE(test_utils)
{
  TestLayout layout;
  TestRandom rng;
  BaseDouble test_base(layout.lattice_shape, layout.block_shape);

  bool site_utils_check = true;
  for (int i = 0; i < layout.lattice_volume; ++i) {
    std::vector<int> coords = test_base.get_site_coords(i);
    for (int j = 0; j < NDIM; ++j) {
      if (coords[j] < 0 || coords[j] >= layout.lattice_shape[j]) {
	BOOST_TEST_MESSAGE("Computed site coordinates exceed lattice bounds "
			   "(function get_site_coords(int))");
	site_utils_check = false;
	break;
      }
    }
    if (test_base.get_site_index(coords) != i) {
      BOOST_TEST_MESSAGE("Unable to recreate site index from coordinates from "
			 "get_site_coords(int)");
      site_utils_check = false;
    }
    test_base.get_site_coords(i, coords);
    for (int j = 0; j < NDIM; ++j) {
      if (coords[j] < 0 || coords[j] >= layout.lattice_shape[j]) {
	BOOST_TEST_MESSAGE("Computed site coordinates exceed lattice bounds "
			   "(function get_site_coords(int, std::vector<int>))");
	site_utils_check = false;
	break;
      }
    }
    if (test_base.get_site_index(coords) != i) {
      BOOST_TEST_MESSAGE("Unable to recreate site index from coordinates from "
			 "get_site_coords(int, std::vector<int>)");
      site_utils_check = false;
    }
    if (not site_utils_check)
      break;
  }
  BOOST_REQUIRE(site_utils_check);

  double rand_num = rng.gen_real();
  test_base = rand_num;
  BOOST_TEST_CASE(boost::bind(&const_value_test, test_base, rand_num));
}

BOOST_AUTO_TEST_CASE(test_accessors)
{
  TestLayout layout;
  TestRandom rng;
  
  BaseDouble test_base(layout.lattice_shape, layout.block_shape);

  std::vector<double> random_values(layout.lattice_volume, 0.0);
  bool sqr_bracket_check = true;
  for (int i = 0; i < layout.lattice_volume; ++i) {
    random_values[i] = rng.gen_real();
    test_base[i] = random_values[i];
  }
  for (int i = 0; i < layout.lattice_volume; ++i) {
    if (not fp_compare(test_base[i], random_values[i])) {
      sqr_bracket_check = false;
      break;
    }
  }
  BOOST_CHECK(sqr_bracket_check);

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
  bool parantheses_check = true;
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
    if (not fp_compare(test_base(COORD_INDEX_PARAMS(n)), random_values[i])) {
    parantheses_check = false;
    break;
  }
  }
    BOOST_CHECK(parantheses_check);
}

BOOST_AUTO_TEST_CASE(test_arithmetic)
{
  TestLayout layout;
  TestRandom rng;
  
  double random_1 = rng.gen_real();
  double random_2 = rng.gen_real();
  BaseDouble base_1(random_1, layout.lattice_shape, layout.block_shape);
  BaseDouble base_2(random_2, layout.lattice_shape, layout.block_shape);

  base_1 *= random_2;
  base_2 /= random_1;

  BOOST_TEST_CASE(boost::bind(&const_value_test, base_1,
			      random_1 * random_2));
  BOOST_TEST_CASE(boost::bind(&const_value_test, base_2,
			      random_2 / random_1));

  base_1 = random_1;
  base_2 = random_2;

  base_1 += base_2;
  base_1 = BaseDouble(random_1, layout.lattice_shape, layout.block_shape);
  base_1 -= base_2;

  BOOST_TEST_CASE(boost::bind(&const_value_test, base_1,
			      random_1 + random_2));
  BOOST_TEST_CASE(boost::bind(&const_value_test, base_1,
			      random_1 - random_2));

  BaseDouble base_sum = base_1 + base_2;
  BaseDouble base_diff = base_1 - base_2;
  BaseDouble base_multiple = random_2 * base_1;
  BaseDouble base_multiple_2 = base_1 * random_2;
  BaseDouble base_div = base_1 / random_2;

  BOOST_TEST_CASE(boost::bind(&const_value_test, base_sum,
			      random_1 + random_2));
  BOOST_TEST_CASE(boost::bind(&const_value_test, base_diff,
			      random_1 - random_2));
  BOOST_TEST_CASE(boost::bind(&const_value_test, base_multiple,
			      random_1 * random_2));
  BOOST_TEST_CASE(boost::bind(&const_value_test, base_multiple_2,
			      random_1 * random_2));
  BOOST_TEST_CASE(boost::bind(&const_value_test, base_div,
			      random_1 / random_2));
}

BOOST_AUTO_TEST_SUITE_END()
