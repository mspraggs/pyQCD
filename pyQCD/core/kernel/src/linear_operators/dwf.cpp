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



vector<VectorXcd> DWF::apply(const vector<VectorXcd>& psi)
{
  // Right multiply a vector by the operator

  // The output vector
  vector<VectorXcd> eta \
    = vector<VectorXcd>(this->Ls_, VectorXcd::Zero(this->operatorSize_));

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

  unsigned long long nKernelFlopsOld = this->kernel_->getNumFlops();

  for (int i = 0; i < this->Ls_; ++i) {
    eta[i] = this->kernel_->apply(psi[i]);

    if (i == 0) {
      eta[i] -= pyQCD::multiplyPminus(psi[1]);
      eta[i] += this->mass_ * pyQCD::multiplyPplus(psi[this->Ls_ - 1]);
    }
    else if (i == this->Ls_ - 1) {
      eta[i] -= pyQCD::multiplyPplus(psi[this->Ls_ - 2]);
      eta[i] += this->mass_ * pyQCD::multiplyPminus(psi[0]);
    }
    else {
      eta[i] -= pyQCD::multiplyPminus(psi[i + 1]);
      eta[i] -= pyQCD::multiplyPplus(psi[i - 1]);
    }
  }

  this->nFlops_ += this->kernel_->getNumFlops() - nKernelFlopsOld;

  return eta;
}



vector<VectorXcd> DWF::applyHermitian(const vector<VectorXcd>& psi)
{
  return this->makeHermitian(this->apply(psi));
}



vector<VectorXcd> DWF::makeHermitian(const vector<VectorXcd>& psi)
{
  // Right multiply a vector by the operator daggered
  // The output vector
  vector<VectorXcd> eta \
    = vector<VectorXcd>(this->Ls_, VectorXcd::Zero(this->operatorSize_));

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

  unsigned long long nKernelFlopsOld = this->kernel_->getNumFlops();

  for (int i = 0; i < this->Ls_; ++i) {
    eta[i] = pyQCD::multiplyGamma5(psi[i]);
    eta[i] = this->kernel_->apply(eta[i]);
    eta[i] = pyQCD::multiplyGamma5(psi[i]);

    if (i == 0) {
      eta[i] -= pyQCD::multiplyPplus(psi[1]);
      eta[i] += this->mass_ * pyQCD::multiplyPminus(psi[this->Ls_ - 1]);
    }
    else if (i == this->Ls_ - 1) {
      eta[i] -= pyQCD::multiplyPminus(psi[this->Ls_ - 2]);
      eta[i] += this->mass_ * pyQCD::multiplyPplus(psi[0]);
    }
    else {
      eta[i] -= pyQCD::multiplyPplus(psi[i + 1]);
      eta[i] -= pyQCD::multiplyPminus(psi[i - 1]);
    }
  }

  this->nFlops_ += this->kernel_->getNumFlops() - nKernelFlopsOld;

  return eta;
}


