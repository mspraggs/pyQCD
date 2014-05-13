#ifndef CUDA_HAMBER_WU_H
#define CUDA_HAMBER_WU_H

#include <linear_operator.h>

#include <utils.h>
#include <kernels.h>

class CudaHamberWu : public CudaLinearOperator
{
public:
  CudaHamberWu(const float mass, const int L, const int T, const bool precondition,
	   const bool hermitian, const Complex* boundaryConditions,
	   Complex* links, const bool copyLinks);
  ~CudaHamberWu();

  void apply(Complex* y, const Complex* x) const;
  void applyHermitian(Complex* y, const Complex* x) const;
  void makeHermitian(Complex* y, const Complex* x) const;

private:

  float mass_;

  CudaLinearOperator* nearestHopping_;
  CudaLinearOperator* nextNearestHopping_;
};

#include <hamber_wu.tcu>

#endif
