#ifndef CUDA_WILSON_H
#define CUDA_WILSON_H

#include <linear_operator.h>

#include <utils.h>
#include <hopping_term.h>

class CudaWilson : public CudaLinearOperator
{
public:
  CudaWilson(const float mass, const int L, const int T, const bool precondition,
	     const bool hermitian, const Complex* boundaryConditions,
	     Complex* links, const bool copyLinks);
  ~CudaWilson();

  void apply(Complex* y, const Complex* x) const;
  void applyHermitian(Complex* y, const Complex* x) const;
  void makeHermitian(Complex* y, const Complex* x) const;

  void applyEvenEvenInv(Complex* y, const Complex* x) const;
  void applyOddOdd(Complex* y, const Complex* x) const;
  void applyEvenOdd(Complex* y, const Complex* x) const;
  void applyOddEven(Complex* y, const Complex* x) const;

  void applyPreconditionedHermitian(Complex* y, const Complex* x) const;

private:
  float mass_;
  CudaLinearOperator* hoppingTerm_;
};

#include <wilson.tcu>

#endif
