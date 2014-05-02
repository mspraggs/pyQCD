#ifndef CUDA_NAIK_H
#define CUDA_NAIK_H

#include <utils.h>
#include <kernels.h>

#include <linear_operator.h>

class CudaNaik : public CudaLinearOperator
{
public:
  CudaNaik(const float mass, const int L, const int T, const bool precondition,
       const bool hermitian, const Complex* boundaryConditions,
       Complex* links, const bool copyLinks);
  ~CudaNaik();

  void apply(Complex* y, const Complex* x) const;
  void applyHermitian(Complex* y, const Complex* x) const;
  void makeHermitian(Complex* y, const Complex* x) const;

private:

  float mass_;

  int* neighbourIndices_;
  int* nextNextNeighbourIndices_;

  Complex* spinStructures_;
  Complex* naikSpinStructures_;
  Complex* boundaryConditions_;
  Complex* naikBoundaryConditions_;
};

#endif
