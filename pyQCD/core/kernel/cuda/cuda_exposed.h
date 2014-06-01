#ifndef CUDA_EXPOSED_H
#define CUDA_EXPOSED_H

#include <base.h>

// This shadows the exposed functions in cuda_interface.cu. We need
// to used external C linkage, at least for now, because for some
// reason the cuda name mangling doesn't match g++ name mangling.

extern "C"
{
  namespace pyQCD
  {
   
    void invertDiracOperator(VectorTypeHost& psi, const int action,
			     const int* intParams, const float* floatParams,
			     const Complex* complexParams,
			     const Complex* boundaryConditions,
			     const VectorTypeHost& eta, const int solverMethod,
			     const int precondition, int* maxIterations,
			     double* tolerance, const int verbosity,
			     const int L, const int T,
			     const Complex* gaugeField);
  
    void computePropagator(PropagatorTypeHost& result,
			   const int action, const int* intParams,
			   const float* floatParams,
			   const Complex* complexParams,
			   const Complex* boundaryConditions, const int site[4],
			   const int sourceSmearingType, const int nSourceSmears,
			   const float sourceSmearingParameter,
			   const int sinkSmearingType, const int nSinkSmears,
			   const float sinkSmearingParameter,
			   const int solverMethod, const int maxIterations,
			   const float tolerance, const int precondition,
			   const int verbosity, const int L, const int T,
			   const Complex* gaugeField);
  }
}
#endif
