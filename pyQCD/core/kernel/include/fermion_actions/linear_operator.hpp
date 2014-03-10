#ifndef LINEAR_OPERATOR_HPP
#define LINEAR_OPERATOR_HPP

#include <complex>
#include <vector>

#include <Eigen/Dense>

#include <omp.h>

#include <lattice.hpp>

using namespace Eigen;
using namespace std;

class LinearOperator
{
public:
  LinearOperator();
  
  VectorXcd apply(const VectorXcd& x);
  VectorXcd apply_hermitian(const VectorXcd& x);
  VectorXcd undo_hermiticity(const VectorXcd& x);
}
