#ifndef CUDA_HAMBER_WU_H
#define CUDA_HAMBER_WU_H

#include <linear_operator.h>

#include <utils.h>
#include <kernels.h>

class HamberWu : public LinearOperator
{
public:
  HamberWu(const float mass, const int L, const int T, const bool precondition,
	   const bool hermitian, const Complex* boundaryConditions,
	   Complex* links, const bool copyLinks);
  ~HamberWu();

  void apply(Complex* y, const Complex* x) const;
  void applyHermitian(Complex* y, const Complex* x) const;
  void makeHermitian(Complex* y, const Complex* x) const;

private:

  float mass_;

  int* neighbourIndices_;
  int* nextNeighbourIndices_;

  Complex* spinStructures_;
  Complex* hamberWuSpinStructures_;
  Complex* boundaryConditions_;
  Complex* hamberWuBoundaryConditions_;
};

#endif
