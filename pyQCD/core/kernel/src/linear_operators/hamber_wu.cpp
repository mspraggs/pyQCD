#include <linear_operators/hamber_wu.hpp>

HamberWu::HamberWu(
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

  vector<Matrix4cd> hamberWuSpinStructures;

  for (int i = 0; i < 4; ++i) {
    hamberWuSpinStructures.push_back(2 * Matrix4cd::Identity()
				     - pyQCD::gammas[i]);
  }

  for (int i = 0; i < 4; ++i) {
    hamberWuSpinStructures.push_back(2 * Matrix4cd::Identity() 
				     + pyQCD::gammas[i]);
  }

  this->nextNearestNeighbour_ = new HoppingTerm(boundaryConditions,
						hamberWuSpinStructures,
						lattice, 2);
}



HamberWu::~HamberWu()
{
  // Just the hopping matrix to destroy
  delete this->nearestNeighbour_;
  delete this->nextNearestNeighbour_;
}



VectorXcd HamberWu::apply(const VectorXcd& psi)
{
  // Right multiply a vector by the operator
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

  eta = (1 + 3 / this->lattice_->chi() + this->mass_) * psi;

  this->nFlops_ += 6 * this->operatorSize_;

  unsigned long long nearestFlopsOld = this->nearestNeighbour_->getNumFlops();
  unsigned long long nextNearestFlopsOld
    = this->nextNearestNeighbour_->getNumFlops();

  // Apply the derivative component
  eta += 2.0 / 3.0 * this->nearestNeighbour_->apply(psi);
  eta -= 1.0 / 12.0 * this->nextNearestNeighbour_->apply(psi);

  this->nFlops_ += this->nearestNeighbour_->getNumFlops() - nearestFlopsOld;
  this->nFlops_
    += this->nextNearestNeighbour_->getNumFlops() - nextNearestFlopsOld;

  return eta;
}



VectorXcd HamberWu::applyHermitian(const VectorXcd& psi)
{
  VectorXcd eta = this->apply(psi);

  return pyQCD::multiplyGamma5(eta);
}



VectorXcd HamberWu::makeHermitian(const VectorXcd& psi)
{
  return pyQCD::multiplyGamma5(psi);
}


