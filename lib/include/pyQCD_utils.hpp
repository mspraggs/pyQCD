#ifndef PYQCD_UTILS_HPP
#define PYQCD_UTILS_HPP

#include <Eigen/Dense>
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
  extern const Matrix4cd gammas[5];

  extern mt19937 generator;
  extern uniform_real<> uniformFloat;
  extern uniform_int<> uniformInt;
  extern variate_generator<mt19937&, uniform_real<> > uni;
  extern variate_generator<mt19937&, uniform_int<> > randomIndex;

  Matrix4cd gamma(const int index);

  int mod(int number, const int divisor);
  int sgn(const int x);
  void getLinkIndices(int n, int link[5]);

  void createSu2(Matrix2cd& out, const double coefficients[4]);
  void embedSu2(const Matrix2cd& Su2Matrix, Matrix3cd& Su3Matrix,
		const int index);
  void extractSubMatrix(const Matrix3cd& su3Matrix, Matrix2cd& subMatrix,
			const int index);
  void extractSu2(const Matrix3cd& su3Matrix, Matrix2cd& su2Matrix,
		  double coefficients[4], const int index);  
}

#endif
