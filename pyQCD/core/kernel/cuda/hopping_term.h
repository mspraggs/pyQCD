#ifndef CUDA_HOPPING_TERM_H
#define CUDA_HOPPING_TERM_H

#include <linear_operator.h>

#include <utils.h>
#include <kernels.h>

template <int numHops>
class CudaHoppingTerm : public CudaLinearOperator
{
public:
  CudaHoppingTerm(const Complex scaling, const int L, const int T,
		  const bool precondition, const bool hermitian,
		  const Complex* boundaryConditions,
		  Complex* links, const bool copyLinks);
  CudaHoppingTerm(const Complex scaling, const int L, const int T,
		  const bool precondition, const bool hermitian,
		  const Complex* boundaryConditions,
		  const Complex* spinStructures, const int spinLength,
		  Complex* links, const bool copyLinks);
  ~CudaHoppingTerm();

  void init(const Complex scaling, const int L, const int T,
	    const bool precondition, const bool hermitian,
	    const Complex* boundaryConditions, 
	    const Complex* spinStructures, Complex* links,
	    const bool copyLinks);

  void apply3d(Complex* y, const Complex* x) const;
  void apply(Complex* y, const Complex* x) const;
  void applyHermitian(Complex* y, const Complex* x) const;
  void makeHermitian(Complex* y, const Complex* x) const;

  void applyEvenOdd(Complex* y, const Complex* x) const;
  void applyOddEven(Complex* y, const Complex* x) const;

private:
  
  Complex* spinStructures_;
  int* neighbours_;
  Complex* boundaryConditions_;
  Complex scaling_;
}

#include <hopping_term.tcu>

#endif
