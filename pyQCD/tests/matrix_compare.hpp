#ifndef PYQCD_MATRIX_COMPARE_HPP
#define PYQCD_MATRIX_COMPARE_HPP

#include <boost/test/unit_test.hpp>

template <typename MatrixType>
struct MatrixCompare
{
  MatrixCompare(const double percent_tolerance)
    : _percent_tolerance(percent_tolerance)
  { }
  
  bool operator()(const MatrixType& rhs, const MatrixType& lhs) const
  {
    return ((rhs.array() - lhs.array()).abs()
	    > _percent_tolerance * rhs.array().abs()).any()
      || ((rhs.array() - lhs.array()).abs()
	  > _percent_tolerance * lhs.array().abs()).any();
  }
  
  double _percent_tolerance;
};

#endif
