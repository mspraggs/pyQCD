#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE base_test
#include <boost/test/unit_test.hpp>

#include <vector>
#include <exception>

#include <base/lattice_base.hpp>

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
  ~TestLayout()
  { 
    BOOST_TEST_MESSAGE("Tear down test layout");
  };

  std::vector<int> lattice_shape;
  std::vector<int> bad_latt_shape;
  std::vector<int> block_shape;
  int lattice_volume;
  int block_volume;
  int num_blocks;
};

BOOST_AUTO_TEST_SUITE(test_lattice_base)

BOOST_AUTO_TEST_CASE(constructors)
{
  typedef pyQCD::LatticeBase<double> Base;
  TestLayout layout;
  Base test_latt_base(layout.lattice_shape, layout.block_shape);
  BOOST_CHECK_EQUAL_COLLECTIONS(test_latt_base.lattice_shape().begin(),
				test_latt_base.lattice_shape().end(),
				layout.lattice_shape.begin(),
				layout.lattice_shape.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(test_latt_base.block_shape().begin(),
				test_latt_base.block_shape().end(),
				layout.block_shape.begin(),
				layout.block_shape.end());
  BOOST_CHECK_EQUAL(test_latt_base.layout().size(), layout.lattice_volume);
  BOOST_CHECK_EQUAL(test_latt_base.lattice_volume(), layout.lattice_volume);
  BOOST_CHECK_EQUAL(test_latt_base.block_volume(), layout.block_volume);
  BOOST_CHECK_EQUAL(test_latt_base.num_blocks(), layout.num_blocks);

  BOOST_CHECK_THROW(Base test_latt_base_bad(layout.bad_latt_shape,
					    layout.block_shape),
		    std::invalid_argument);

  Base base_1(1.0, layout.lattice_shape, layout.block_shape);
  Base base_2(2.0, layout.lattice_shape, layout.block_shape);
  Base base_sum = base_1 + base_2;
  for (int i = 0; i < base_1.lattice_volume(); ++i)
    BOOST_CHECK_CLOSE(base_sum[i], 3.0, 0.0001);
}

BOOST_AUTO_TEST_SUITE_END()
