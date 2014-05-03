#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H

#include <cusp/linear_operator.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/bicgstab.h>

#include <utils.h>
#include <wilson.h>
#include <jacobi_smearing.h>
#include <hamber_wu.h>
#include <naik.h>
#include <dwf.h>

namespace pyQCD
{
  void makeSource(VectorTypeDev& eta, const int site[4], const int spin,
		  const int colour, const CudaLinearOperator* smearingOperator);

  void diracOperatorFactory(CudaLinearOperator* diracOperator, const int action,
			    const int* intParams, const float* floatParams,
			    const Complex* complexParams,
			    const Complex* boundaryConditions,
			    const int L, const int T, const bool precondition,
			    const bool hermitian, const Complex* links,
			    const bool copyLinks);
}
#endif
