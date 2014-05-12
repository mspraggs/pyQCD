#include <linear_operators/linear_operator.hpp>

VectorXcd LinearOperator::makeEvenOdd(const VectorXcd& x)
{
  // Permutes the supplied spinor, shuffling it so the upper half
  // contains the even sites and the lower half contains the odd sites

  VectorXcd y = VectorXcd::Zero(this->operatorSize_);

  if (x.size() != this->operatorSize_)
    return y;

  int nSites = this->operatorSize_ / 12;

  if (this->evenIndices_.size() != nSites / 2
      || this->oddIndices_.size() != nSites / 2)
    return y;

#pragma omp parallel for
  for (int i = 0; i < nSites / 2; ++i)
    for (int j = 0; j < 12; ++j)
      y(12 * i + j) = x(12 * this->evenIndices_[i] + j);

#pragma omp parallel for
  for (int i = nSites / 2; i < nSites; ++i)
    for (int j = 0; j < 12; ++j)
      y(12 * i + j) = x(12 * this->oddIndices_[i - nSites / 2] + j);
  
  return y;
}



VectorXcd LinearOperator::removeEvenOdd(const VectorXcd& x)
{
  // Permutes the supplied spinor, shuffling it so that the upper half
  // and lower halves and reordered into lexicographic order.

  VectorXcd y = VectorXcd::Zero(this->operatorSize_);

  if (x.size() != this->operatorSize_)
    return y;

  int nSites = this->operatorSize_ / 12;

  if (this->evenIndices_.size() != nSites / 2
      || this->oddIndices_.size() != nSites / 2)
    return y;

#pragma omp parallel for
  for (int i = 0; i < nSites / 2; ++i)
    for (int j = 0; j < 12; ++j)
      y(12 * this->evenIndices_[i] + j) = x(12 * i + j);

#pragma omp parallel for
  for (int i = nSites / 2; i < nSites; ++i)
    for (int j = 0; j < 12; ++j)
      y(12 * this->oddIndices_[i - nSites / 2] + j) = x(12 * i + j);

  return y;
}



VectorXcd LinearOperator::makeEvenOddSource(const VectorXcd& x)
{
  // Create the source required to an even-odd inversion

  VectorXcd y = VectorXcd::Zero(this->operatorSize_);

  if (x.size() != this->operatorSize_)
    return y;

  int N = this->operatorSize_ / 2;

  y = this->makeEvenOdd(x);
  y.tail(N) -= this->applyOddEven(this->applyEvenEvenInv(x.head(N)));

  return y;
}



VectorXcd LinearOperator::makeEvenOddSolution(const VectorXcd& x)
{
  // Create the standard lexicographic solution from the solution to
  // the even-odd-preconditioned equation.

  VectorXcd y = x;

  if (x.size() != this->operatorSize_)
    return y;

  int N = this->operatorSize_ / 2;

  y.head(N) -= this->applyEvenEvenInv(this->applyEvenOdd(x.tail(N)));

  return this->removeEvenOdd(y);
}
