#ifndef PYUTILS_HPP
#define PYUTILS_HPP

#include <vector>

#include <Eigen/Dense>
#include <boost/python.hpp>
#include <boost/python/list.hpp>

using namespace boost;
namespace py = boost::python;
using namespace Eigen;
using namespace std;

namespace pyQCD
{
  void propagatorPrep(int site[4], vector<complex<double> >& boundaryConditions,
		      const py::list& pySite,
		      const py::list& pyBoundaryConditions);

  py::list convertMatrixToList(const MatrixXcd& matrix);
  MatrixXcd convertListToMatrix(const py::list list);
  py::list convertVectorToList(const VectorXcd& vector);
  VectorXcd convertListToVector(const py::list list);
  py::list propagatorToList(vector<MatrixXcd>& propagator);
}

#endif
