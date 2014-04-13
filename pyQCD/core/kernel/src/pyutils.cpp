#include <pyutils.hpp>

namespace pyQCD
{
  void propagatorPrep(int site[4], vector<complex<double> >& boundaryConditions,
		      const py::list& pySite,
		      const py::list& pyBoundaryConditions)
  { 
    for (int i = 0; i < 4; ++i)
      site[i] = py::extract<int>(pySite[i]);

    boundaryConditions = vector<complex<double> >(4, complex<double>(1.0, 0.0));
    
    for (int i = 0; i < 4; ++i)
      boundaryConditions[i] 
	= py::extract<complex<double> >(pyBoundaryConditions[i]);
  }

  vector<complex<double> > convertBoundaryConditions(
    const py::list& boundaryConditions)
  {
    out = vector<complex<double> >(4, complex<double>(1.0, 0.0));
    
    for (int i = 0; i < 4; ++i)
      out[i] = py::extract<complex<double> >(boundaryConditions[i]);

    return out
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



  MatrixXcd convertListToMatrix(const py::list list)
  {
    int nRows = py::len(list);
    int nCols = py::len(list[0]);
    
    MatrixXcd matrixOut(nRows, nCols);

    for (int i = 0; i < nRows; ++i)
      for (int j = 0; j < nCols; ++j)
	matrixOut(i, j) = py::extract<complex<double> >(list[i][j]);

    return matrixOut;
  }



  py::list convertVectorToList(const VectorXcd& vector)
  {
    int nRows = vector.size();
    
    py::list listOut;
    
    for (int i = 0; i < nRows; ++i) {
      listOut.append(vector(i));
    }
    return listOut;
  }



  VectorXcd convertListToVector(const py::list list)
  {
    int nRows = py::len(list);
    
    VectorXcd vectorOut(nRows);

    for (int i = 0; i < nRows; ++i)
	vectorOut(i) = py::extract<complex<double> >(list[i]);

    return vectorOut;
  }

  py::list propagatorToList(vector<MatrixXcd>& prop)
  {
    // This is where we'll store the propagator as a list
    py::list pythonPropagator;
    
    int nSites = prop.size();
    // Loop through the raw propagator and add it to the python list
    for (int i = 0; i < nSites; ++i) {
      py::list matrixList = convertMatrixToList(prop[i]);
      pythonPropagator.append(matrixList);
    }

    return pythonPropagator;
  }
}
