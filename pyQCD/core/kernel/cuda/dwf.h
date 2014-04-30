#ifndef CUDA_DWF_H
#define CUDA_DWF_H

#include <utils.h>
#include <kernels.h>

#include <linear_operator.h>
#include <wilson.h>
#include <hamber_wu.h>
#include <naik.h>

class DWF : public LinearOperator
{
public:
  DWF(const float mass, const float M5, const int Ls, const int kernelType,
      const int L, const int T, const bool precondition, const bool hermitian,
      const Complex boundaryConditions[4], Complex* links);
  ~DWF();

  void apply(Complex* y, const Complex* x) const;
  void applyHermitian(Complex* y, const Complex* x) const;
  void makeHermitian(Complex* y, const Complex* x) const;

private:
  float mass_;
  float M5_;
  int Ls_;

  LinearOperator* kernel_;
};

#endif
