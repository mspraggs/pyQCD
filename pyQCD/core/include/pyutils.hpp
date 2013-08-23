#ifndef PYUTILS_HPP
#define PYUTILS_HPP

#include <Eigen/Dense>
#include <boost/python.hpp>
#include <boost/python/list.hpp>

using namespace boost;
namespace py = boost::python;
using namespace Eigen;
using namespace std;

namespace pyQCD
{
  py::list convertMatrixToList(const MatrixXcd& matrix);
  MatrixXcd convertListToMatrix(const py::list list);
}

#endif
