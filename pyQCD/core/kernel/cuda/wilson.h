#ifndef CUDA_WILSON_H
#define CUDA_WILSON_H

#include <linear_operator.h>

#include <utils.h>
#include <kernels.h>

class Wilson : public LinearOperator
{
public:
  Wilson(const float mass, const int L, const int T, const bool precondition,
	 const bool hermitian, const Complex* boundaryConditions,
	 Complex* links, const bool copyLinks);
  ~Wilson();

  void apply(Complex* y, const Complex* x) const;
  void applyHermitian(Complex* y, const Complex* x) const;
  void makeHermitian(Complex* y, const Complex* x) const;

private:
  float mass_;

  int* neighbourIndices_;

  Complex* spinStructures_;
  Complex* boundaryConditions_;
};

#endif