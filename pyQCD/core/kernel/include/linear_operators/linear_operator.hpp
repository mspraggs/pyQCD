#ifndef LINEAR_OPERATOR_HPP
#define LINEAR_OPERATOR_HPP

#include <Eigen/Dense>

#include <omp.h>

#include <linear_operators/utils.hpp>

using namespace Eigen;
using namespace std;

class Lattice;

class LinearOperator
{
  // The base class from which all Dirac operators are derived
public:
  LinearOperator() { nFlops_ = 0; };
  virtual ~LinearOperator() { };
  
  // Applies the linear operator to a column vector using right multplication
  virtual VectorXcd apply(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }
  // Applies a hermition form of the the linear operator to a column vector
  // using right multiplication
  virtual VectorXcd applyHermitian(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }
  // Undoes the hermiticity operation applied by applyHermitian. Note that
  // undoHermiticity(applyHermitian(x)) = apply(x)
  virtual VectorXcd undoHermiticity(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }

  unsigned long long getNumFlops() { return this->nFlops_; }

protected:
  unsigned long long nFlops_;
};

#endif
