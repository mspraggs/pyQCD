#include <linear_operators/linear_operator.hpp>

VectorXcd LinearOperator::forwardSubstitute(const VectorXcd& b)
{
  // Solve L*x = b using forward substitution

  VectorXcd x = VectorXcd::Zero(this->operatorSize_);

  if (b.size() != this->operatorSize_)
    return x;

  // First get the diagonal elements
  VectorXcd diagonal = this->applyDiagonal(VectorXcd::Ones(this->operatorSize_));

  for (int i = 0; i < this->operatorSize_; ++i) {
    x[i] = b[i];
    x[i] -= this->lowerRowDot(x, i);
    x[i] /= diagonal(i);
  }

  return x;
}



VectorXcd LinearOperator::backSubstitute(const VectorXcd& b)
{
  // Solve L*x = b using forward substitution

  VectorXcd x = VectorXcd::Zero(this->operatorSize_);

  if (b.size() != this->operatorSize_)
    return x;

  // First get the diagonal elements
  VectorXcd diagonal = this->applyDiagonal(VectorXcd::Ones(this->operatorSize_));

  for (int i = this->operatorSize_ - 1; i >= 0; --i) {
    x[i] = b[i];
    x[i] -= this->upperRowDot(x, i);
    x[i] /= diagonal(i);
  }

  return x;
}
