#include <linear_operator.h>

LinearOperator::LinearOperator(const int L, const int T, const int precondition,
			       const int hermitian, Complex* links,
			       const bool copyLinks)
  : super(12 * L * L * L * T, 12 * L * L * L * T)
{
  this->N = 12 * L * L * L * T;
  this->L = L;
  this->T = T;

  this->precondition_ = precondition;
  this->hermitian_ = hermitian;

  // Number of complex numbers in the array of links
  if (copyLinks) {
    int size = 3 * this->N * sizeof(Complex);
    cudaMalloc((void**) &this->links_, size);
    cudaMemcpy(this->links_, links, size, cudaMemcpyHostToDevice);
  }
  else
    this->links_ = links;
}



LinearOperator::~LinearOperator()
{
  cudaFree(this->links_);
}



void LinearOperator::operator()(const VectorTypeDev& x, VectorTypeDev& y) const
{
  const Complex* x_ptr = thrust::raw_pointer_cast(&x[0]);
  Complex* y_ptr = thrust::raw_pointer_cast(&y[0]);

  if (this->precondition_) {
    
  }
  else {
    if (this->hermitian_)
      this->applyHermitian(y_ptr, x_ptr);
    else
      this->apply(y_ptr, x_ptr);
  }
}

