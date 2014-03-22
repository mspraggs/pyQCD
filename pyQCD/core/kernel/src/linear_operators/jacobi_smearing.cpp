#include <linear_operators/jacobi_smearing.hpp>

JacobiSmearing::JacobiSmearing(
    const int numSmears, const double smearingParameter,
    const vector<complex<double> >& boundaryConditions,
    Lattice* lattice) : LinearOperator::LinearOperator()
{
  // Class constructor - we set the pointer to the lattice and determine
  // the nearest neighbours

  this->operatorSize_ 
    = 12 * int(pow(lattice->spatialExtent, 3)) * lattice->temporalExtent;

  this->smearingParameter_ = smearingParameter;
  this->numSmears_ = numSmears;
  this->lattice_ = lattice;

  this->hoppingMatrix_ = new WilsonHoppingTerm(boundaryConditions,
					       Matrix4cd::Identity(), lattice);

  // Initialise tadpole coefficients
  this->tadpoleCoefficients_[0] = lattice->ut();
  this->tadpoleCoefficients_[1] = lattice->us();
  this->tadpoleCoefficients_[2] = lattice->us();
  this->tadpoleCoefficients_[3] = lattice->us();

  this->nearestNeighbours_ = pyQCD::getNeighbourIndices(1, this->lattice_);
  this->boundaryConditions_ = pyQCD::getBoundaryConditions(1, boundaryConditions,
							   this->lattice_);
}



JacobiSmearing::~JacobiSmearing()
{
  // Delete the hopping matrix
  delete this->hoppingMatrix_;
}



VectorXcd JacobiSmearing::apply(const VectorXcd& psi)
{
  // Apply the smearing operator itself

  VectorXcd eta = psi; // The output

  // Temporary vector to use in the sum
  VectorXcd tempTerm = VectorXcd::Zero(psi.size());

  for (int i = 0; i < this->numSmears_; ++i) {
    tempTerm = this->smearingParameter_ * this->applyOnce(tempTerm);
    eta += tempTerm;
  }

  return eta;
}



VectorXcd JacobiSmearing::applyOnce(const VectorXcd& psi)
{
  // Right multiply a vector once by the operator H (see Jacobi smearing in
  // Gattringer and Lang).
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

  eta = this->hoppingMatrix_->apply3d(eta);

  return eta;
}



VectorXcd JacobiSmearing::applyHermitian(const VectorXcd& psi)
{
  // Right multiply a vector by the operator
  return this->apply(psi);
}



VectorXcd JacobiSmearing::undoHermiticity(const VectorXcd& psi)
{
  // Undo Hermticity in applyHermtian (there is none, so retur psi)
  return psi;
}
