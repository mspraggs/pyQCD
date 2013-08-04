#include <pyQCD_utils.hpp>

namespace pyQCD
{
  const complex<double> i (0.0, 1.0);
  const double pi = 3.1415926535897932384626433;


  const Matrix2cd sigma0 = Matrix2cd::Identity();

  const Matrix2cd sigma1 = (MatrixXcd(2, 2) << 0, 1,
			    1, 0).finished();

  const Matrix2cd sigma2 = (MatrixXcd(2, 2) << 0, -i,
			    i, 0).finished();

  const Matrix2cd sigma3 = (MatrixXcd(2, 2) << 1, 0,
			    0, -1).finished();

  const Matrix2cd sigmas[4] = {sigma0, sigma1, sigma2, sigma3};



  const Matrix4cd gamma0 = (MatrixXcd(4, 4) << 0, 0, 1, 0,
			    0, 0, 0, 1,
			    1, 0, 0, 0,
			    0, 1, 0, 0).finished();

  const Matrix4cd gamma1 = (MatrixXcd(4, 4) << 0, 0, 0, -i,
			    0, 0, -i, 0,
			    0, i, 0, 0,
			    i, 0, 0, 0).finished();
  
  const Matrix4cd gamma2 = (MatrixXcd(4, 4) <<  0, 0, 0, -1,
			    0, 0, 1, 0,
			    0, 1, 0, 0,
			    -1, 0, 0, 0).finished();

  const Matrix4cd gamma3 = (MatrixXcd(4, 4) << 0, 0, -i, 0,
			    0, 0, 0, i,
			    i, 0, 0, 0,
			    0, -i, 0, 0).finished();

  const Matrix4cd gamma4 = gamma0;

  const Matrix4cd gamma5 = (MatrixXcd(4, 4) << 1, 0, 0, 0,
			    0, 1, 0, 0,
			    0, 0, -1, 0,
			    0, 0, 0, -1).finished();
  
  const Matrix4cd gammas[6] = {gamma0, gamma1, gamma2, gamma3, gamma4, gamma5};



  mt19937 generator(time(0));
  uniform_real<> uniformFloat(0, 1);
  uniform_int<> uniformInt(0, 199);
  variate_generator<mt19937&, uniform_real<> > uni(generator,
						   uniformFloat);
  variate_generator<mt19937&, uniform_int<> > randomIndex(generator,
							  uniformInt);



  Matrix4cd gamma(const int index)
  {
    int prefactor = sgn(index);
    return prefactor * gammas[abs(index) - 1];
  }



  int mod(int number, const int divisor)
  {
    int ret = number % divisor;
    if (ret < 0)
      ret += divisor;
    return ret;
  }



  int sgn(const int x)
  {
    return (x < 0) ? -1 : 1;
  }



  void getLinkIndices(int n, const int size, int link[5])
  {
    int i = 0;
    link[4] = mod(n, 4);
    n /= 4;
    for (int i = 0; i < 4; ++i) {
      link[3 - i] = mod(n, size);
      n /= size;
    }
  }



  int getLinkIndex(const int link[5], const int size)
  {
    return link[4] + 4
      * (link[3] + size
	 * (link[2] + size
	    * (link[1] + size
	       * link[0])));
  }



  int getLinkIndex(const int n0, const int n1, const int n2, const int n3,
		   const int n4, const int size)
  {
    return n4 + 4 * (n3 + size * (n2 + size * (n1 + size * n0)));
  }



  Matrix2cd createSu2(const double coefficients[4])
  {
    Matrix2cd out = coefficients[0] * sigmas[0];
    for (int j = 1; j < 4; ++j)
      out += i * coefficients[j] * sigmas[j];
    return out;
  }



  Matrix3cd embedSu2(const Matrix2cd& Su2Matrix,
		const int index)
  {
    Matrix3cd Su3Matrix;
    if (index == 0) {
      Su3Matrix(0, 0) = 1.0;
      Su3Matrix(0, 1) = 0.0;
      Su3Matrix(0, 2) = 0.0;
      Su3Matrix(1, 0) = 0.0;
      Su3Matrix(1, 1) = Su2Matrix(0, 0);
      Su3Matrix(1, 2) = Su2Matrix(0, 1);
      Su3Matrix(2, 0) = 0.0;
      Su3Matrix(2, 1) = Su2Matrix(1, 0);
      Su3Matrix(2, 2) = Su2Matrix(1, 1);
    }
    else if (index == 1) {
      Su3Matrix(0, 0) = Su2Matrix(0, 0);
      Su3Matrix(0, 1) = 0.0;
      Su3Matrix(0, 2) = Su2Matrix(0, 1);
      Su3Matrix(1, 0) = 0.0;
      Su3Matrix(1, 1) = 1.0;
      Su3Matrix(1, 2) = 0.0;
      Su3Matrix(2, 0) = Su2Matrix(1, 0);
      Su3Matrix(2, 1) = 0.0;
      Su3Matrix(2, 2) = Su2Matrix(1, 1);
    }
    else {    
      Su3Matrix(0, 0) = Su2Matrix(0, 0);
      Su3Matrix(0, 1) = Su2Matrix(0, 1);
      Su3Matrix(0, 2) = 0.0;
      Su3Matrix(1, 0) = Su2Matrix(1, 0);
      Su3Matrix(1, 1) = Su2Matrix(1, 1);
      Su3Matrix(1, 2) = 0.0;
      Su3Matrix(2, 0) = 0.0;
      Su3Matrix(2, 1) = 0.0;
      Su3Matrix(2, 2) = 1.0;
    }
    return Su3Matrix;
  }



  Matrix2cd extractSubMatrix(const Matrix3cd& su3Matrix,
			const int index)
  {
    Matrix2cd subMatrix;
    if (index == 0) {
      subMatrix(0, 0) = su3Matrix(1, 1);
      subMatrix(0, 1) = su3Matrix(1, 2);
      subMatrix(1, 0) = su3Matrix(2, 1);
      subMatrix(1, 1) = su3Matrix(2, 2);
    }
    else if (index == 1) {
      subMatrix(0, 0) = su3Matrix(0, 0);
      subMatrix(0, 1) = su3Matrix(0, 2);
      subMatrix(1, 0) = su3Matrix(2, 0);
      subMatrix(1, 1) = su3Matrix(2, 2);
    }
    else {
      subMatrix(0, 0) = su3Matrix(0, 0);
      subMatrix(0, 1) = su3Matrix(0, 1);
      subMatrix(1, 0) = su3Matrix(1, 0);
      subMatrix(1, 1) = su3Matrix(1, 1);
    }
    return subMatrix;
  }



  Matrix2cd extractSu2(const Matrix3cd& su3Matrix,
		  double coefficients[4], const int index)
  {
    Matrix2cd subMatrix = extractSubMatrix(su3Matrix, index);
    
    coefficients[0] = subMatrix(0, 0).real() + subMatrix(1, 1).real();
    coefficients[1] = subMatrix(0, 1).imag() + subMatrix(1, 0).imag();
    coefficients[2] = subMatrix(0, 1).real() - subMatrix(1, 0).real();
    coefficients[3] = subMatrix(0, 0).imag() - subMatrix(1, 1).imag();

    double magnitude = sqrt(pow(coefficients[0], 2) +
			    pow(coefficients[1], 2) +
			    pow(coefficients[2], 2) +
			    pow(coefficients[3], 2));

    return createSu2(coefficients) / magnitude;
  }



  py::list convertMatrixToList(const MatrixXcd& matrix)
  {
    int nRows = matrix.rows();
    int nCols = matrix.cols();
    
    py::list listOut;
    
    for (int i = 0; i < nRows; ++i) {
      py::list tempList;
      for (int j = 0; j < nCols; ++j) {
	tempList.append(matrix(i,j));
      }
      listOut.append(tempList);
    }
    return listOut;
  }

  

  double oneNorm(const Matrix3cd& matrix)
  {
    double out = 0.0;
    
    for (int i = 0; i < matrix.rows(); ++i) {
      for (int j = 0; j < matrix.cols(); ++j) {
	out += norm(matrix(i,j));
      }
    }

    return out;
  }
}
