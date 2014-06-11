
CudaHamberWu::CudaHamberWu(const float mass, const int L, const int T,
		   const bool precondition, const bool hermitian,
		   const Complex* boundaryConditions, Complex* links,
		   const bool copyLinks = true)
  : CudaLinearOperator(L, T, precondition, hermitian, links, copyLinks)
{
  this->mass_ = mass;

  this->nearestHopping_ = new CudaHoppingTerm<1>(-2.0 / 3.0, L, T, precondition,
						 hermitian, boundaryConditions,
						 this->links_, false);

  Complex hostSpinStructures[128];
  Complex hostGammas[64];
  createGammas(hostGammas);

  // Now set the spin structes up with the identity, which we'll then subtract
  // and add the gamma matrices from/to.
  diagonalSpinMatrices(hostSpinStructures, 2.0);
  diagonalSpinMatrices(hostSpinStructures + 64, 2.0);
  subtractArray(hostSpinStructures, hostGammas, 64); // 2 - gamma_mu
  addArray(hostSpinStructures + 64, hostGammas, 64); // 2 + gamma_mu

  this->nextNearestHopping_ \
    = new CudaHoppingTerm<2>(1.0 / 12.0, L, T, precondition,
			     hermitian, boundaryConditions,
			     hostSpinStructures, 128,
			     this->links_, false);
}



CudaHamberWu::~CudaHamberWu()
{
  delete this->nearestHopping_;
  delete this->nextNearestHopping_;
}



void CudaHamberWu::apply(Complex* y, const Complex* x) const
{  
  int dimBlock;
  int dimGrid;

  setGridAndBlockSize(dimBlock, dimGrid, this->N);

  diagonalKernel<<<dimGrid,dimBlock>>>(y, x, 4 + this->mass_,
				       this->L_, this->T_);

  this->nearestHopping_->apply(y, x);
  this->nextNearestHopping_->apply(y, x);
}



void CudaHamberWu::applyHermitian(Complex* y, const Complex* x) const
{
  this->apply(y, x);
  int dimBlock;
  int dimGrid;

  setGridAndBlockSize(dimBlock, dimGrid, this->N);
  applyGamma5<<<dimGrid,dimBlock>>>(y, y, this->L_, this->T_);
}



void CudaHamberWu::makeHermitian(Complex* y, const Complex* x) const
{
  int dimBlock;
  int dimGrid;

  int divisor = this->precondition_ ? 2 : 1;

  setGridAndBlockSize(dimBlock, dimGrid, this->N / divisor);
  applyGamma5<<<dimGrid,dimBlock>>>(y, x, this->L_, this->T_ / divisor);
}
