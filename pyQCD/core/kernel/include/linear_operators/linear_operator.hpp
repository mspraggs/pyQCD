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

  const unsigned long long getNumFlops() const { return this->nFlops_; }
  const vector<int>& getEvenIndices() const { return this->evenIndices_; }
  const vector<int>& getOddIndices() const { return this->oddIndices_; }

  // Functions to set things up for the even-odd preconditioning
  virtual VectorXcd makeEvenOdd(const VectorXcd& x);
  virtual VectorXcd removeEvenOdd(const VectorXcd& x);
  VectorXcd makeEvenOddSource(const VectorXcd& x);
  VectorXcd makeEvenOddSolution(const VectorXcd& x);

  // Even-odd apply functions
  virtual VectorXcd applyEvenEvenInv(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }
  virtual VectorXcd applyOddOdd(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }
  virtual VectorXcd applyEvenOdd(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }
  virtual VectorXcd applyOddEven(const VectorXcd& x)
  { return VectorXcd::Zero(x.size()); }
  VectorXcd applyPreconditioned(const VectorXcd& x)
  {
    return this->applyOddOdd(x)
      - this->applyOddEven(this->applyEvenEvenInv(this->applyEvenOdd(x)));
  }
  VectorXcd applyPreconditionedHermitian(const VectorXcd& x)
  { return this->makeHermitian(this->applyPreconditioned(x)); }

protected:
  unsigned long long nFlops_;
  int operatorSize_; // Size of vectors on which the operator may operate

  vector<int> evenIndices_;
  vector<int> oddIndices_;
  vector<vector<int> > evenNeighbours_;
  vector<vector<int> > oddNeighbours_;
};

#endif
