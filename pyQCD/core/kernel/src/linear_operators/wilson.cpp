#include <linear_operators/wilson.hpp>

Wilson::Wilson(
    const double mass, const vector<complex<double> >& boundaryConditions,
    Lattice* lattice) : LinearOperator::LinearOperator()
{
  // Class constructor - we set the fermion mass, create a pointer to the 
  // lattice and compute the frequently used spin structures used within the
  // Dirac operator.
  this->mass_ = mass;
  this->operatorSize_ 
    = 12 * int(pow(lattice->spatialExtent, 3)) * lattice->temporalExtent;
  this->lattice_ = lattice;

  this->hoppingMatrix_ = new HoppingTerm(boundaryConditions, lattice, 1);
}



Wilson::~Wilson()
{
  // Just the hopping matrix to destroy
  delete this->hoppingMatrix_;
}



VectorXcd Wilson::apply(const VectorXcd& psi)
{
  // Right multiply a vector by the operator
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

  eta = (1 + 3 / this->lattice_->chi() + this->mass_) * psi;

  this->nFlops_ += 6 * this->operatorSize_;

  unsigned long long nHoppingFlopsOld = this->hoppingMatrix_->getNumFlops();

  // Apply the derivative component
  eta -= 0.5 * this->hoppingMatrix_->apply(psi);

  this->nFlops_ += this->hoppingMatrix_->getNumFlops() - nHoppingFlopsOld;

  return eta;
}



VectorXcd Wilson::applyHermitian(const VectorXcd& psi)
{
  VectorXcd eta = this->apply(psi);

  return pyQCD::multiplyGamma5(eta);
}



VectorXcd Wilson::makeHermitian(const VectorXcd& psi)
{
  return pyQCD::multiplyGamma5(psi);
}



complex<double> Wilson::lowerRowDot(const VectorXcd& psi, const int row)
{
  return -this->hoppingMatrix_->lowerRowDot(psi, row);
}



complex<double> Wilson::upperRowDot(const VectorXcd& psi, const int row)
{
  return -this->hoppingMatrix_->upperRowDot(psi, row);
}



VectorXcd Wilson::applyDiagonal(const VectorXcd& psi)
{
  return (1 + 3 / this->lattice_->chi() + this->mass_) * psi;
}
