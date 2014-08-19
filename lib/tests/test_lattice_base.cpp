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

boost::test_tools::close_at_tolerance<double>
fp_compare(boost::test_tools::percent_tolerance(1e-8));

class BaseDouble : public pyQCD::LatticeBase<double, NDIM>
{
public:
  using pyQCD::LatticeBase<double, NDIM>::LatticeBase;
  using pyQCD::LatticeBase<double, NDIM>::operator=;
  // Member access functions
  const std::vector<int>& block_shape() const
  { return this->_block_shape; }
  const std::vector<std::vector<int> >& layout() const
  { return this->_layout; }
  const int num_blocks() const
  { return this->_num_blocks; }
  const int block_volume() const
  { return this->_block_volume; }
  const double datum_ref(const int i, const int j) const
  { 
    assert(i > -1 && i < this->_num_blocks);
    assert(j > -1 && j < this->_block_volume);
    return this->_data[i][j];
  }
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



void const_value_test(const BaseDouble& lattice_base, const double value,
		      const int start = 0, const int end = -1)
{
  bool all_values_equal = true;
  int true_end = (end < 0) ? lattice_base.num_blocks() : end;
  for (int i = start; i < true_end; ++i)
    for (int j = 0; j < lattice_base.block_volume(); ++j)
      if (not fp_compare(lattice_base.datum_ref(i, j), value)) {
	all_values_equal = false;
	break;
      }
  BOOST_CHECK(all_values_equal);
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
    = boost::bind(&const_value_test, _1, rand_num, 0, -1);
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
  BOOST_TEST_CASE(boost::bind(&const_value_test, test_base, rand_num, 0, -1));
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
    random_values[i] = rng.gen_real();
    test_base[coords] = random_values[i];
  }
  bool parantheses_check = true;
  for (int i = 0; i < layout.lattice_volume; ++i) {
    std::vector<int> coords = test_base.get_site_coords(i);
    if (not fp_compare(test_base[coords], random_values[i])) {
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

  const_value_test(base_1, random_1 * random_2, 0, -1);
  const_value_test(base_2, random_2 / random_1, 0, -1);

  base_1 = random_1;
  base_2 = random_2;

  base_1 += base_2;
  const_value_test(base_1, random_1 + random_2, 0, -1);
  base_1 = BaseDouble(random_1, layout.lattice_shape, layout.block_shape);
  base_1 -= base_2;
  const_value_test(base_1, random_1 - random_2, 0, -1);

  base_1 = random_1;
  base_2 = random_2;

  BaseDouble base_sum = base_1 + base_2;
  BaseDouble base_diff = base_1 - base_2;
  BaseDouble base_multiple = random_2 * base_1;
  BaseDouble base_multiple_2 = base_1 * random_2;
  BaseDouble base_div = base_1 / random_2;

  const_value_test(base_sum, random_1 + random_2, 0, -1);
  const_value_test(base_diff, random_1 - random_2, 0, -1);
  const_value_test(base_multiple, random_1 * random_2, 0, -1);
  const_value_test(base_multiple_2, random_1 * random_2, 0, -1);
  const_value_test(base_div, random_1 / random_2, 0, -1);

  BaseDouble base_sum_mult = random_1 * (base_1 + base_2);
  BaseDouble base_mult_sum = random_1 * base_1 + base_2;
  BaseDouble base_sum_sum = base_1 + (base_1 + base_2);
  const_value_test(base_sum_mult, random_1 * (random_1 + random_2),
		   0, -1);
  const_value_test(base_mult_sum, random_1 * random_1 + random_2,
		   0, -1);
  const_value_test(base_sum_sum, 2 * random_1 + random_2,
		   0, -1);
}

BOOST_AUTO_TEST_CASE(test_expressions)
{
  TestLayout layout;
  TestRandom rng;
  
  double random_1 = rng.gen_real();
  double random_2 = rng.gen_real();
  BaseDouble base_1(random_1, layout.lattice_shape, layout.block_shape);
  BaseDouble base_2(random_2, layout.lattice_shape, layout.block_shape);

  base_1.even_sites() = base_2.even_sites();
  const_value_test(base_1, random_2, 0, base_1.num_blocks() / 2);
  const_value_test(base_1, random_1, base_1.num_blocks() / 2,
		   base_1.num_blocks());

  base_1.even_sites() = (2.0 * base_2 + base_2).even_sites();
  const_value_test(base_1, 3 * random_2, 0, base_1.num_blocks() / 2);
  const_value_test(base_1, random_1, base_1.num_blocks() / 2,
		   base_1.num_blocks());

  base_1 = random_1;
  base_2 = random_2;

  base_1.odd_sites() = base_2.odd_sites();
  const_value_test(base_1, random_1, 0, base_1.num_blocks() / 2);
  const_value_test(base_1, random_2, base_1.num_blocks() / 2,
		   base_1.num_blocks());

  base_1.odd_sites() = (2.0 * base_2 + base_2).odd_sites();
  const_value_test(base_1, random_1, 0, base_1.num_blocks() / 2);
  const_value_test(base_1, 3 * random_2, base_1.num_blocks() / 2,
		   base_1.num_blocks());

  for (int i = 0; i < base_1.lattice_volume(); ++i)
    base_1[i] = rng.gen_real();
  BaseDouble base_roll = base_1.roll(2, -1);

  bool roll_equal = true;
  for (int i = 0; i < base_1.lattice_volume(); ++i) {
    std::vector<int> site = base_1.get_site_coords(i);
    std::vector<int> site_roll = site;
    site_roll[2] = (site_roll[2] + 1) % base_1.lattice_shape()[2];

    if (not fp_compare(base_1[site_roll], base_roll[site])) {
      BOOST_TEST_MESSAGE("Index 1: " << i << " Index roll: "
			 << base_1.get_site_index(site_roll));
      roll_equal = false;
      break;
    }
  }
  BOOST_CHECK(roll_equal);
}

BOOST_AUTO_TEST_SUITE_END()
