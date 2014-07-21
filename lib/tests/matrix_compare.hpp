
#include <boost/test/unit_test.hpp>

template <typename MatrixType>
struct MatrixCompare
{
  MatrixCompare(const double percent_tolerance)
    : _percent_tolerance(percent_tolerance)
  {
    BOOST_TEST_MESSAGE("Set up matrix comparison");
  }
  ~MatrixCompare()
  { BOOST_TEST_MESSAGE("Tear down matrix comparison"); }
  
  bool operator(const MatrixType& rhs, const MatrixType& lhs) const
  {
    bool result = true;
    return (rhs.array() - lhs.array()).abs()
      > _percent_tolerance * rhs.array().abs()
      || (rhs.array() - lhs.array()).abs()
      > _percent_tolerance * lhs.array().abs();
  }
  
  double _percent_tolerance;
};
