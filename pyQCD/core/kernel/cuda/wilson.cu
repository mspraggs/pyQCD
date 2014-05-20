
CudaWilson::CudaWilson(const float mass, const int L, const int T,
	       const bool precondition, const bool hermitian,
	       const Complex boundaryConditions[4], Complex* links,
	       const bool copyLinks)
  : CudaLinearOperator(L, T, precondition, hermitian, links, copyLinks)
{
  this->mass_ = mass;

  this->hoppingTerm_ = new CudaHoppingTerm<1>(-0.5, L, T, precondition,
					      hermitian, boundaryConditions,
					      this->links_, false);

  this->evenIndices_ = this->hoppingTerm_->getEvenIndices();
  this->oddIndices_ = this->hoppingTerm_->getOddIndices();
}



CudaWilson::~CudaWilson()
{
  delete this->hoppingTerm_;
}



void CudaWilson::apply(Complex* y, const Complex* x) const
{
  
  int dimBlock;
  int dimGrid;

  setGridAndBlockSize(dimBlock, dimGrid, this->N);

  diagonalKernel<<<dimGrid,dimBlock>>>(y, x, 4 + this->mass_,
				       this->L_, this->T_);

  this->hoppingTerm_->apply(y, x);
}



void CudaWilson::applyHermitian(Complex* y, const Complex* x) const
{
  this->apply(y, x);
  int dimBlock;
  int dimGrid;
  setGridAndBlockSize(dimBlock, dimGrid, this->N);
  applyGamma5<<<dimGrid,dimBlock>>>(y, y, this->L_, this->T_);
}



void CudaWilson::makeHermitian(Complex* y, const Complex* x) const
{
  int dimBlock;
  int dimGrid;

  int divisor = this->precondition_ ? 2 : 1;

  setGridAndBlockSize(dimBlock, dimGrid, this->N / divisor);
  applyGamma5<<<dimGrid,dimBlock>>>(y, x, this->L_, this->T_ / divisor);
}



void CudaWilson::applyEvenEvenInv(Complex* y, const Complex* x) const
{
  int dimBlock;
  int dimGrid;
  setGridAndBlockSize(dimBlock, dimGrid, this->N / 2);
  
  diagonalKernel<<<dimGrid,dimBlock>>>(y, x, 1.0 / (4.0 + this->mass_),
				       this->L_, this->T_ / 2);
}



void CudaWilson::applyOddOdd(Complex* y, const Complex* x) const
{
  int dimBlock;
  int dimGrid;
  setGridAndBlockSize(dimBlock, dimGrid, this->N / 2);
  
  diagonalKernel<<<dimGrid,dimBlock>>>(y, x, 4.0 + this->mass_,
				       this->L_, this->T_ / 2);
}



void CudaWilson::applyEvenOdd(Complex* y, const Complex* x) const
{
  this->hoppingTerm_->applyEvenOdd(y, x);
}



void CudaWilson::applyOddEven(Complex* y, const Complex* x) const
{
  this->hoppingTerm_->applyOddEven(y, x);
}



void CudaWilson::applyPreconditionedHermitian(Complex* y, const Complex* x) const
{
  this->applyPreconditioned(y, x);
  int dimBlock;
  int dimGrid;
  setGridAndBlockSize(dimBlock, dimGrid, this->N / 2);
  applyGamma5<<<dimGrid,dimBlock>>>(y, y, this->L_, this->T_ / 2);
}
