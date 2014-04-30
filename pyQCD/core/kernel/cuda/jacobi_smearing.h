#ifndef CUDA_JACOBI_SMEARING_H
#define CUDA_JACOBI_SMEARING_H

#include <linear_operator.h>

#include <utils.h>
#include <kernels.h>

class JacobiSmearing : public LinearOperator
{
public:
  JacobiSmearing(const int nSmears, const double smearingParameter,
		 const int L, const int T, const Complex* boundaryConditions,
		 Complex* links, const bool copyLinks);
  ~JacobiSmearing();

  void applyOnce(Complex* y, const Complex* x) const;
  void apply(Complex* y, const Complex* x) const;

private:
  int nSmears_;
  double smearingParameter_;

  int* neighbourIndices_;

  Complex* spinStructures_;
  Complex* boundaryConditions_;
};

#endif
