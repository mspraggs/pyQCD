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
   
  void invertDiracOperator(VectorTypeHost& psi, const VectorTypeHost& eta,
			   CudaLinearOperator* diracMatrix, const int solverMethod,
			   const int precondition, const int maxIterations,
			   const double tolerance, const int verbosity);
  
  void computePropagator(PropagatorTypeHost& result,
			 const CudaLinearOperator* diracMatrix, const int site[4],
			 const int sourceSmearingType, const int nSourceSmears,
			 const float sourceSmearingParameter,
			 const int sinkSmearingType, const int nSinkSmears,
			 const float sinkSmearingParameter,
			 const int solverMethod, const int maxIterations,
			 const float tolerance, const int precondition,
			 const int verbosity);
}
#endif
