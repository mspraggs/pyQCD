#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_site_gauge_field

#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>

#include <base/site_gauge_field.hpp>

#include "random.hpp"
#include "compare_matrix.hpp"

typedef Eigen::Matrix3cd ColourType
typedef pyQCD::SiteGaugeField<ColourType> Field;

void constructor_test(const Field& gauge_field)
{
  
}

BOOST_AUTO_TEST_SUITE(test_site_gauge_field)

BOOST_AUTO_TEST_CASE(test_constructors)
{
  TestRandom rng;
  MatrixCompare<ColourType> matrix_compare(1e-8);
  Field::colour_type rand_col_mat = Field::colour_type::Random();

  std::vector<Field> gauge_fields;
  gauge_fields.push_back(Field());
  gauge_fields.push_back(Field(rand_col_mat));
  gauge_fields.push_back(Field(gauge_fields[1]));
  Field test_equals = gauge_fields[1];
  gauge_fields.push_back(test_equals);
  
  BOOST_PARAM_TEST_CASE(constructor_test,
			gauge_fields.begin(), gauge_fields.end());
}

BOOST_AUTO_TEST_CASE(test_utils)
{
  
}

BOOST_AUTO_TEST_CASE(test_accessors)
{
  
}

BOOST_AUTO_TEST_CASE(test_arithmetic)
{
  
}

BOOST_AUTO_TEST_SUITE_END()
