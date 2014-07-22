#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_site_gauge_field

#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>

#include <base/site_gauge_field.hpp>

#include "random.hpp"
#include "matrix_compare.hpp"

typedef Eigen::Matrix3cd ColourType
typedef pyQCD::SiteGaugeField<ColourType> Field;

MatrixCompare<ColourType> matrix_compare(1e-8);



struct FieldExposed : public Field
{
  
}



void constructor_test(const FieldExposed& gauge_field)
{
  BOOST_REQUIRE_EQUAL(gauge_field.data().size(), NDIM)
}



void const_value_test(const FieldExposed& gauge_field,
		      const ColourType& value)
{
  bool all_vals_equal = true;
  for (int i = 0; i < NDIM; ++i)
    if (not matrix_compare(gauge_field.data()[i], value)) {
      BOOST_TEST_MESSAGE("All elements in SiteGaugeField not equal to supplied "
			 "value");
      all_vals_equal = false;
      break;
    }
  BOOST_CHECK(all_vals_equal);
}



BOOST_AUTO_TEST_SUITE(test_site_gauge_field)

BOOST_AUTO_TEST_CASE(test_constructors)
{
  TestRandom rng;
  ColourType rand_col_mat = ColourType::Random();

  std::vector<FieldExposed> gauge_fields;
  gauge_fields.push_back(FieldExposed());
  gauge_fields.push_back(FieldExposed(rand_col_mat));
  gauge_fields.push_back(FieldExposed(gauge_fields[1]));
  FieldExposed test_equals = gauge_fields[1];
  gauge_fields.push_back(test_equals);
  
  BOOST_PARAM_TEST_CASE(constructor_test,
			gauge_fields.begin(), gauge_fields.end());

  boost::unit_test::callback1<BaseDouble> bound_const_value_test
    = boost::bind(&const_value_test, _1, rand_col_mat);
  BOOST_PARAM_TEST_CASE(bound_const_value_test,
			gauge_fields.begin() + 1, gauge_fields.end());
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
