#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <boost/random.hpp>

using namespace boost;
using namespace Eigen;
using namespace std;

namespace pyQCD
{
  extern const complex<double> i;
  extern const double pi;


  extern const Matrix2cd sigma0;
  extern const Matrix2cd sigma1;
  extern const Matrix2cd sigma2;
  extern const Matrix2cd sigma3;
  extern const Matrix2cd sigmas[4];
  extern const Matrix4cd gamma1;
  extern const Matrix4cd gamma2;
  extern const Matrix4cd gamma3;
  extern const Matrix4cd gamma4;
  extern const Matrix4cd gamma5;
  extern const Matrix4cd gammas[6];

  extern mt19937 generator;
  extern uniform_real<> uniformFloat;
  extern uniform_int<> uniformInt;
  extern variate_generator<mt19937&, uniform_real<> > uni;
  extern variate_generator<mt19937&, uniform_int<> > randomIndex;

  Matrix4cd gamma(const int index);

  int mod(int number, const int divisor);
  int sgn(const int x);
  void getLinkIndices(int n, const int spaceSize, const int timeSize,
		      int link[5]);
  int getLinkIndex(const int link[5], const int size);
  int getLinkIndex(const int n0, const int n1, const int n2, const int n3,
		   const int n4, const int size);

  Matrix2cd createSu2(const double coefficients[4]);
  Matrix3cd embedSu2(const Matrix2cd& Su2Matrix,
		     const int index);
  Matrix2cd extractSubMatrix(const Matrix3cd& su3Matrix,
			     const int index);
  Matrix2cd extractSu2(const Matrix3cd& su3Matrix,
		       double coefficients[4], const int index);

  double oneNorm(const Matrix3cd& matrix);
}

#endif
