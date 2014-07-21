#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_site_gauge_field

#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>

#include <base/site_gauge_field.hpp>

BOOST_AUTO_TEST_SUITE(test_site_gauge_field)

typedef pyQCD::SiteGaugeField Field

BOOST_AUTO_TEST_CASE(test_constructors)
{
  std::vector<Field> gauge_fields;
  gauge_fields.push_back(Field());
  gauge_fields.push_back(Field(gauge_fields[0]));
}

BOOST_AUTO_TEST_SUITE_END()
