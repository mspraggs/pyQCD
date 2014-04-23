#include <linear_operators/naik.hpp>

Naik::Naik(
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

  this->nearestNeighbour_ = new HoppingTerm(boundaryConditions, lattice, 1);

  // Create the Hamber-Wu spin structures

  vector<Matrix4cd> naikSpinStructures;

  for (int i = 0; i < 4; ++i) {
    naikSpinStructures.push_back(3 * Matrix4cd::Identity()
				     - pyQCD::gammas[i]);
  }

  for (int i = 0; i < 4; ++i) {
    naikSpinStructures.push_back(3 * Matrix4cd::Identity() 
				     + pyQCD::gammas[i]);
  }

  this->nextNextNearestNeighbour_ = new HoppingTerm(boundaryConditions,
						    naikSpinStructures,
						    lattice, 3);

  this->evenIndices_ = this->nearestNeighbour_->getEvenIndices();
  this->oddIndices_ = this->nearestNeighbour_->getOddIndices();
}



Naik::~Naik()
{
  // Just the hopping matrix to destroy
  delete this->nearestNeighbour_;
  delete this->nextNextNearestNeighbour_;
}



VectorXcd Naik::apply(const VectorXcd& psi)
{
  // Right multiply a vector by the operator
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

  eta = (1 + 3 / this->lattice_->chi() + this->mass_) * psi;

  this->nFlops_ += 6 * this->operatorSize_;

  unsigned long long nearestFlopsOld = this->nearestNeighbour_->getNumFlops();
  unsigned long long nextNextNearestFlopsOld
    = this->nextNextNearestNeighbour_->getNumFlops();

  // Apply the derivative component
  eta -= 2.0 / 3.0 * this->nearestNeighbour_->apply(psi);
  eta += 2.0 / 81.0 * this->nextNextNearestNeighbour_->apply(psi);

  this->nFlops_ += this->nearestNeighbour_->getNumFlops() - nearestFlopsOld;
  this->nFlops_
    += this->nextNextNearestNeighbour_->getNumFlops() - nextNextNearestFlopsOld;

  return eta;
}



VectorXcd Naik::applyHermitian(const VectorXcd& psi)
{
  VectorXcd eta = this->apply(psi);

  return pyQCD::multiplyGamma5(eta);
}



VectorXcd Naik::makeHermitian(const VectorXcd& psi)
{
  return pyQCD::multiplyGamma5(psi);
}



VectorXcd Naik::applyEvenEvenInv(const VectorXcd& psi)
{
  // Invert the even diagonal piece
  return psi / (1 + 3 / this->lattice_->chi() + this->mass_);
}



VectorXcd Naik::applyOddOdd(const VectorXcd& psi)
{
  // Invert the even diagonal piece
  return (1 + 3 / this->lattice_->chi() + this->mass_) * psi;
}



VectorXcd Naik::applyEvenOdd(const VectorXcd& psi)
{
  return 2.0 / 81.0 * this->nextNextNearestNeighbour_->applyEvenOdd(psi)
    - 2.0 / 3.0 * this->nearestNeighbour_->applyEvenOdd(psi);
}



VectorXcd Naik::applyOddEven(const VectorXcd& psi)
{
  return 2.0 / 81.0 * this->nextNextNearestNeighbour_->applyOddEven(psi)
    - 2.0 / 3.0 * this->nearestNeighbour_->applyOddEven(psi);
}


