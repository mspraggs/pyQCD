
CudaNaik::CudaNaik(const float mass, const int L, const int T,
	   const bool precondition, const bool hermitian,
	   const Complex* boundaryConditions, Complex* links,
	   const bool copyLinks = true)
  : CudaLinearOperator(L, T, precondition, hermitian, links, copyLinks)
{
  this->mass_ = mass;

  this->nearestHopping_ = new CudaHoppingTerm<1>(-9.0 / 16.0, L, T,
						 precondition, hermitian,
						 boundaryConditions,
						 this->links_, false);

  Complex hostSpinStructures[128];
  Complex hostGammas[64];
  createGammas(hostGammas);

  // Now set the spin structes up with the identity, which we'll then subtract
  // and add the gamma matrices from/to.
  diagonalSpinMatrices(hostSpinStructures, Complex(3.0, 0.0));
  diagonalSpinMatrices(hostSpinStructures + 64, Complex(3.0, 0.0));
  subtractArray(hostSpinStructures, hostGammas, 64); // 3 - gamma_mu
  addArray(hostSpinStructures + 64, hostGammas, 64); // 3 + gamma_mu

  this->nextNextNearestHopping_ \
    = new CudaHoppingTerm<3>(1.0 / 6.0, L, T, precondition,
			     hermitian, boundaryConditions,
			     this->links_, false);

  this->evenIndices_ = this->nearestHopping_->getEvenIndices();
  this->oddIndices_ = this->nearestHopping_->getOddIndices();
}



CudaNaik::~CudaNaik()
{
  delete this->nearestHopping_;
  delete this->nextNextNearestHopping_;
}



void CudaNaik::apply(Complex* y, const Complex* x) const
{  
  int dimBlock;
  int dimGrid;

  setGridAndBlockSize(dimBlock, dimGrid, this->N);

  diagonalKernel<<<dimGrid,dimBlock>>>(y, x, 4 + this->mass_,
				       this->L_, this->T_);

  this->nearestHopping_->apply(y, x);
  this->nextNextNearestHopping_->apply(y, x);
}



void CudaNaik::applyHermitian(Complex* y, const Complex* x) const
{
  this->apply(y, x);
  int dimBlock;
  int dimGrid;
  setGridAndBlockSize(dimBlock, dimGrid, this->N);
  applyGamma5<<<dimGrid,dimBlock>>>(y, y, this->L_, this->T_);
}



void CudaNaik::makeHermitian(Complex* y, const Complex* x) const
{
  int dimBlock;
  int dimGrid;

  int divisor = this->precondition_ ? 2 : 1;

  setGridAndBlockSize(dimBlock, dimGrid, this->N / divisor);
  applyGamma5<<<dimGrid,dimBlock>>>(y, x, this->L_, this->T_ / divisor);
}



void CudaNaik::applyEvenEvenInv(Complex* y, const Complex* x) const
{
  int dimBlock;
  int dimGrid;
  setGridAndBlockSize(dimBlock, dimGrid, this->N / 2);
  
  diagonalKernel<<<dimGrid,dimBlock>>>(y, x, 1.0 / (4.0 + this->mass_),
				       this->L_, this->T_ / 2);
}



void CudaNaik::applyOddOdd(Complex* y, const Complex* x) const
{
  int dimBlock;
  int dimGrid;
  setGridAndBlockSize(dimBlock, dimGrid, this->N / 2);
  
  diagonalKernel<<<dimGrid,dimBlock>>>(y, x, 4.0 + this->mass_,
				       this->L_, this->T_ / 2);
}



void CudaNaik::applyEvenOdd(Complex* y, const Complex* x) const
{
  this->nearestHopping_->applyEvenOdd(y, x);
  this->nextNextNearestHopping_->applyEvenOdd(y, x);
}



void CudaNaik::applyOddEven(Complex* y, const Complex* x) const
{
  this->nearestHopping_->applyOddEven(y, x);
  this->nextNextNearestHopping_->applyOddEven(y, x);
}



void CudaNaik::applyPreconditionedHermitian(Complex* y, const Complex* x) const
{
  this->applyPreconditioned(y, x);
  int dimBlock;
  int dimGrid;
  setGridAndBlockSize(dimBlock, dimGrid, this->N / 2);
  applyGamma5<<<dimGrid,dimBlock>>>(y, y, this->L_, this->T_ / 2);
}

