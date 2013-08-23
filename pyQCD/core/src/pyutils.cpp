#include <pyutils.hpp>

namespace pyQCD
{
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
}
