#include <linear_operators/dwf.hpp>

DWF::DWF(
  const double mass, const double M5, const int Ls, const int kernelType,
  const vector<complex<double> >& boundaryConditions,
  Lattice* lattice) : LinearOperator::LinearOperator()
{
  // Class constructor - we set the fermion mass, create a pointer to the 
  // lattice and compute the frequently used spin structures used within the
  // Dirac operator.
  this->mass_ = mass;
  this->Ls_ = Ls;
  this->operatorSize_ 
    = 12 * int(pow(lattice->spatialExtent, 3)) * lattice->temporalExtent;
  this->lattice_ = lattice;

  if (kernelType == pyQCD::wilson)
    this->kernel_ = new Wilson(-M5, boundaryConditions, lattice);
  else if (kernelType == pyQCD::hamberWu)
    this->kernel_ = new HamberWu(-M5, boundaryConditions, lattice);
  else
    this->kernel_ = new Wilson(-M5, boundaryConditions, lattice);    
}



DWF::~DWF()
{
  // Just the hopping matrix to destroy
  delete this->kernel_;
}



VectorXcd DWF::apply(const VectorXcd& psi)
{
  // Right multiply a vector by the operator

  int size4d = this->operatorSize_ / this->Ls_;

  // The output vector
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_);

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

  unsigned long long nKernelFlopsOld = this->kernel_->getNumFlops();

  for (int i = 0; i < this->Ls_; ++i) {
    eta.segment(i * size4d, size4d)
      = this->kernel_->apply(psi.segment(i * size4d, size4d));

    if (i == 0) {
      eta.segment(i * size4d, size4d)
	-= pyQCD::multiplyPminus(psi.segment(size4d, size4d));
      eta.segment(i * size4d, size4d) 
	+= this->mass_
	* pyQCD::multiplyPplus(psi.segment((this->Ls_ - 1) * size4d, size4d));
    }
    else if (i == this->Ls_ - 1) {
      eta.segment(i * size4d, size4d)
	-= pyQCD::multiplyPplus(psi.segment((this->Ls_ - 2) * size4d, size4d));
      eta.segment(i * size4d, size4d)
	+= this->mass_ * pyQCD::multiplyPminus(psi.segment(0, size4d));
    }
    else {
      eta.segment(i * size4d, size4d)
	-= pyQCD::multiplyPminus(psi.segment((i + 1) * size4d, size4d));
      eta.segment(i * size4d, size4d)
	-= pyQCD::multiplyPplus(psi.segment((i - 1) * size4d, size4d));
    }
  }

  this->nFlops_ += this->kernel_->getNumFlops() - nKernelFlopsOld;

  return eta;
}



VectorXcd DWF::applyHermitian(const VectorXcd& psi)
{
  return this->makeHermitian(this->apply(psi));
}



VectorXcd DWF::makeHermitian(const VectorXcd& psi)
{
  // Right multiply a vector by the operator daggered

  int size4d = this->operatorSize_ / this->Ls_;

  // The output vector
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_);

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

  unsigned long long nKernelFlopsOld = this->kernel_->getNumFlops();

  for (int i = 0; i < this->Ls_; ++i) {
    eta.segment(i * size4d, size4d)
      = pyQCD::multiplyGamma5(psi.segment(i * size4d, size4d));
    eta.segment(i * size4d, size4d)
      = this->kernel_->apply(eta.segment(i * size4d, size4d));
    eta.segment(i * size4d, size4d)
      = pyQCD::multiplyGamma5(eta.segment(i * size4d, size4d));

    if (i == 0) {
      eta.segment(i * size4d, size4d)
	-= pyQCD::multiplyPplus(psi.segment(size4d, size4d));
      eta.segment(i * size4d, size4d) 
	+= this->mass_
	* pyQCD::multiplyPminus(psi.segment((this->Ls_ - 1) * size4d, size4d));
    }
    else if (i == this->Ls_ - 1) {
      eta.segment(i * size4d, size4d)
	-= pyQCD::multiplyPminus(psi.segment((this->Ls_ - 2) * size4d, size4d));
      eta.segment(i * size4d, size4d)
	+= this->mass_ * pyQCD::multiplyPplus(psi.segment(0, size4d));
    }
    else {
      eta.segment(i * size4d, size4d)
	-= pyQCD::multiplyPplus(psi.segment((i + 1) * size4d, size4d));
      eta.segment(i * size4d, size4d)
	-= pyQCD::multiplyPminus(psi.segment((i - 1) * size4d, size4d));
    }
  }

  this->nFlops_ += this->kernel_->getNumFlops() - nKernelFlopsOld;

  return eta;
}


