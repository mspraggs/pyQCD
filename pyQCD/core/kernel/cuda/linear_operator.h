#ifndef CUDA_LINEAR_OPERATOR_H
#define CUDA_LINEAR_OPERATOR_H

#include <cusp/linear_operator.h>

#include <utils.h>
#include <kernels.h>

class CudaLinearOperator : public cusp::linear_operator<Complex, cusp::device_memory>
{
public:
  CudaLinearOperator(const int L, const int T, const int precondition,
		     const int hermitian, const, Complex* links,
		     const bool copyLinks);
  virtual ~CudaLinearOperator();

  // The CUSP linear operator application function
  void operator()(const VectorTypeDev& x, VectorTypeDev& y) const;

  //Applies the linear operator to a column vector using right multiplication
  virtual void apply(Complex* y, const Complex* x) const { };
  virtual void applyHermitian(Complex* y, const Complex* x) const { };
  virtual void makeHermitian(Complex* y, const Complex* x) const { };

  // Even-odd preconditioning functions
  void makeEvenOdd(Complex* y, const Complex* x) const;
  void removeEvenOdd(Complex* y, const Complex* x) const;
  void makeEvenOddSource(Complex* y, const Complex* x) const;
  void makeEvenOddSolution(Complex* y, const Complex* x) const;
  virtual void applyEvenEvenInv(Complex* y, const Complex* x) const { }
  virtual void applyOddOdd(Complex* y, const Complex* x) const { }
  virtual void applyEvenOdd(Complex* y, const Complex* x) const { }
  virtual void applyOddEven(Complex* y, const Complex* x) const { }
  void applyPreconditioned(Complex* y, const Complex* x) const;
  void applyPreconditionedHermitian(Complex* y, const Complex* x) const;

  int L() const { return this->L_; }
  int T() const { return this->T_; }
  Complex* links() const { return this->links_; }

protected:
  int N;
  int L_;
  int T_;
  
  bool precondition_;
  bool hermitian_;

  Complex* links_;
};

#include <linear_operator.tcu>

#endif
