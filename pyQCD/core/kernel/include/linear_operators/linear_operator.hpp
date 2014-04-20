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
  // e.g. we might want to invert g5 * D in the equation
  // g5 * D * psi = g5 * eta
  // This function does the g5 * D part, or some other Hermitian form
  virtual VectorXcd applyHermitian(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }
  // Used to make the source used in any inversions correspond to the 
  // possible hermiticity of the operator
  // In the langauge above, this would create the RHS, such as g5 * eta
  virtual VectorXcd makeHermitian(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }

  // These functions are used when doing SSOR preconditioning
  virtual complex<double> lowerRowDot(const VectorXcd& x, const int row)
  { return complex<double>(0.0, 0.0); }
  virtual complex<double> upperRowDot(const VectorXcd& x, const int row)
  { return complex<double>(0.0, 0.0); }
  virtual VectorXcd applyDiagonal(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }
  virtual VectorXcd forwardSubstitute(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }
  virtual VectorXcd backSubstitute(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }
  VectorXcd applySsor(const VectorXcd& x)
  {
    VectorXcd v = this->backSubstitute(x);
    return v + this->forwardSubstitute(x - this->applyDiagonal(v));
  }

  unsigned long long getNumFlops() { return this->nFlops_; }

protected:
  unsigned long long nFlops_;
};

#endif
