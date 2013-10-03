#include <utils.hpp>


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



  void getLinkIndices(int n, const int spaceSize, const int timeSize,
		      int link[5])
  {
    int i = 0;
    link[4] = mod(n, 4);
    n /= 4;
    for (int i = 0; i < 3; ++i) {
      link[3 - i] = mod(n, spaceSize);
      n /= spaceSize;
    }
    link[0] = mod(n, timeSize);
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

#ifdef USE_CUDA

  void eigenToCusp(const SparseMatrix<complex<double> >& eigenMatrix,
		   complexHybridHost& cuspMatrix)
  {
    // Converts a sparse matrix from Eigen's format to Cusp's COO format

    // Get the number of entries, columns and rows
    int nTriplets = eigenMatrix.nonZeros();
    int nRows = eigenMatrix.rows();
    int nCols = eigenMatrix.cols();

    // Declare somewhere to keep the triplets we're going to make
    cusp::array1d<int, cusp::host_memory> rows(nTriplets);
    cusp::array1d<int, cusp::host_memory> cols(nTriplets);
    cusp::array1d<cusp::complex<float>, cusp::host_memory> values(nTriplets);

    // Loop through the non-zero entries and store the positions and values
    // in the triplet
    int index = 0;
    for (int i = 0; i < eigenMatrix.outerSize(); ++i) {
      for (SparseMatrix<complex<double> >::InnerIterator it(eigenMatrix, i);
	     it; ++it) {
	rows[index] = it.row();
	cols[index] = it.col();
	values[index] = cusp::complex<float>(float(it.value().real()),
					 float(it.value().imag()));

	index++;
      }
    }

    // Resize the matrix we're using
    cuspMatrix.resize(nRows, nCols, nTriplets);

    // Assign the various values
    for (int i = 0; i < nTriplets; ++i) {
      cuspMatrix.row_indices[i] = rows[i];
      cuspMatrix.column_indices[i] = cols[i];
      cuspMatrix.values[i] = values[i];
    }
  }



  void cudaBiCGstab(const SparseMatrix<complex<double> >& eigenDirac,
		    const SparseMatrix<complex<double> >& eigenSourceSmear,
		    const SparseMatrix<complex<double> >& eigenSinkSmear,
		    const int spatialIndex, vector<MatrixXcd>& propagator)
  {
    int nTriplets = eigenDirac.nonZeros();
    int nRows = eigenDirac.rows();
    int nCols = eigenDirac.cols();
    int nSites = nRows / 12;

    complexHybridHost cuspDirac;
    complexHybridHost cuspSourceSmear;
    complexHybridHost cuspSinkSmear;

    eigenToCusp(eigenDirac, cuspDirac);
    eigenToCusp(eigenSourceSmear, cuspSourceSmear);
    eigenToCusp(eigenSinkSmear, cuspSinkSmear);

    cusp::array2d<cusp::complex<float>,
		  hostMem> cuspPropagator(nRows, 12,
					  cusp::complex<float>(0, 0));

    cuda::bicgstab(cuspDirac, cuspSourceSmear, cuspSinkSmear, spatialIndex,
		   cuspPropagator);

    for (int i = 0; i < nSites; ++i) {
      for (int j = 0; j < 12; ++j) {
	for (int k = 0; k < 12; ++k) {
	  double x = cuspPropagator(12 * i + j, k).real();
	  double y = cuspPropagator(12 * i + j, k).imag();
	  propagator[i](j, k) = complex<double>(x, y);
	}
      }
    }
  }



  void cudaCG(const SparseMatrix<complex<double> >& eigenDiracDiracAdjoint,
	      const SparseMatrix<complex<double> >& eigenDiracAdjoint,
	      const SparseMatrix<complex<double> >& eigenSourceSmear,
	      const SparseMatrix<complex<double> >& eigenSinkSmear,
	      const int spatialIndex, vector<MatrixXcd>& propagator)
  {
    // Wrapper for the cusp/cuda sparse matrix CG inverter
    int nTriplets = eigenDiracDiracAdjoint.nonZeros();
    int nRows = eigenDiracDiracAdjoint.rows();
    int nCols = eigenDiracDiracAdjoint.cols();
    int nSites = nRows / 12;

    complexHybridHost cuspM;
    complexHybridHost cuspDiracAdjoint;
    complexHybridHost cuspSourceSmear;
    complexHybridHost cuspSinkSmear;

    eigenToCusp(eigenDiracDiracAdjoint, cuspM);
    eigenToCusp(eigenDiracAdjoint, cuspDiracAdjoint);
    eigenToCusp(eigenSourceSmear, cuspSourceSmear);
    eigenToCusp(eigenSinkSmear, cuspSinkSmear);

    cusp::array2d<cusp::complex<float>,
		  hostMem> cuspPropagator(nRows, 12,
					  cusp::complex<float>(0, 0));

    cuda::cg(cuspM, cuspDiracAdjoint, cuspSourceSmear, cuspSinkSmear,
	     spatialIndex, cuspPropagator);

    for (int i = 0; i < nSites; ++i) {
      for (int j = 0; j < 12; ++j) {
	for (int k = 0; k < 12; ++k) {
	  double x = cuspPropagator(12 * i + j, k).real();
	  double y = cuspPropagator(12 * i + j, k).imag();
	  propagator[i](j, k) = complex<double>(x, y);
	}
      }
    }
  }
#endif
}
