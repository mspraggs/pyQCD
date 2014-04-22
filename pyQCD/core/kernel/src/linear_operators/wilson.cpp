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

  this->evenIndices_ = this->hoppingMatrix_->getEvenIndices();
  this->oddIndices_ = this->hoppingMatrix_->getOddIndices();
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



VectorXcd Wilson::applyEvenEvenInv(const VectorXcd& psi)
{
  // Invert the even diagonal piece

  VectorXcd eta = VectorXcd::Zero(this->operatorSize_);

  if (psi.size() != this->operatorSize_)
    return eta;

  eta.head(this->operatorSize_ / 2)
    = psi.head(this->operatorSize_ / 2) 
    / (1 + 3 / this->lattice_->chi() + this->mass_);

  return eta;
}



VectorXcd Wilson::applyOddOdd(const VectorXcd& psi)
{
  // Invert the even diagonal piece

  VectorXcd eta = VectorXcd::Zero(this->operatorSize_);

  if (psi.size() != this->operatorSize_)
    return eta;

  eta.tail(this->operatorSize_ / 2)
    = (1 + 3 / this->lattice_->chi() + this->mass_)
    * psi.tail(this->operatorSize_ / 2);

  return eta;
}



VectorXcd Wilson::applyEvenOdd(const VectorXcd& psi)
{
  return this->hoppingMatrix_->applyEvenOdd(psi);
}



VectorXcd Wilson::applyOddEven(const VectorXcd& psi)
{
  return this->hoppingMatrix_->applyOddEven(psi);
}
