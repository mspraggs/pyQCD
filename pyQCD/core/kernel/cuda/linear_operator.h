#ifndef CUDA_LINEAR_OPERATOR_H
#define CUDA_LINEAR_OPERATOR_H

#include <cusp/linear_operator.h>

#include <utils.h>
#include <kernels.h>

class CudaLinearOperator : public cusp::linear_operator<Complex, cusp::device_memory>
{
public:
  CudaLinearOperator(const int L, const int T, const int precondition,
		 const int hermitian, Complex* links,
		 const bool copyLinks);
  virtual ~CudaLinearOperator();

  int* getEvenIndices() const { return this->evenIndices_; }
  int* getOddIndices() const { return this->oddIndices_; }

  // The CUSP linear operator application function
  void operator()(const VectorTypeDev& x, VectorTypeDev& y) const;

  //Applies the linear operator to a column vector using right multiplication
  virtual void apply(Complex* y, const Complex* x) const { }
  virtual void applyHermitian(Complex* y, const Complex* x) const { }
  virtual void makeHermitian(Complex* y, const Complex* x) const { }

  // Even-odd preconditioning functions
  virtual void makeEvenOdd(Complex* y, const Complex* x) const;
  virtual void removeEvenOdd(Complex* y, const Complex* x) const;
  void makeEvenOddSource(Complex* y, const Complex* xe,
			 const Complex* xo) const;
  void makeEvenOddSolution(Complex* y, const Complex* x,
			   const Complex* xo) const;
  virtual void applyEvenEvenInv(Complex* y, const Complex* x) const { }
  virtual void applyOddOdd(Complex* y, const Complex* x) const { }
  virtual void applyEvenOdd(Complex* y, const Complex* x) const { }
  virtual void applyOddEven(Complex* y, const Complex* x) const { }
  void applyPreconditioned(Complex* y, const Complex* x) const
  {
    Complex* z;
    cudaMalloc((void**) &z, this->N / 2 * sizeof(Complex));
    
    int dimGrid;
    int dimBlock;
    setGridAndBlockSize(dimGrid, dimBlock, this->N / 2);
    assignDev<<<dimGrid,dimBlock>>>(z, 0.0, this->N / 2);

    this->applyEvenOdd(z, x);
    this->applyEvenEvenInv(y, z);
    assignDev<<<dimGrid,dimBlock>>>(z, 0.0, this->N / 2);
    this->applyOddEven(z, y);
    
    this->applyOddOdd(y, x);
    
    saxpyDev<<<dimGrid,dimBlock>>>(y, z, -1.0, this->N / 2);

    cudaFree(z);
  }
  virtual void applyPreconditionedHermitian(Complex* y, const Complex* x) const
  { }

  int L() const { return this->L_; }
  int T() const { return this->T_; }
  Complex* links() const { return this->links_; }

protected:
  int N;
  int L_;
  int T_;
  
  bool precondition_;
  bool hermitian_;

  int* evenIndices_;
  int* oddIndices_;
  int* evenNeighbours_;
  int* oddNeighbours_;

  Complex* links_;
};

#include <linear_operator.tcu>

#endif
