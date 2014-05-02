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
  
    void invertWilsonDiracOperator(VectorTypeHost& psi,
				   const VectorTypeHost& eta,
				   const double mass,
				   const Complex boundaryConditions[4],
				   const int solverMethod,
				   const int precondition,
				   const int maxIterations,
				   const double tolerance,
				   const int verbosity,
				   const Complex* gaugeField,
				   const int L, const int T);
  
    void invertHamberWuDiracOperator(VectorTypeHost& psi,
				     const VectorTypeHost& eta,
				     const double mass,
				     const Complex boundaryConditions[4],
				     const int solverMethod,
				     const int precondition,
				     const int maxIterations,
				     const double tolerance,
				     const int verbosity,
				     const Complex* gaugeField,
				     const int L, const int T);
  
    void invertNaikDiracOperator(VectorTypeHost& psi,
				 const VectorTypeHost& eta,
				 const double mass,
				 const Complex boundaryConditions[4],
				 const int solverMethod,
				 const int precondition,
				 const int maxIterations,
				 const double tolerance,
				 const int verbosity,
				 const Complex* gaugeField,
				 const int L, const int T);

  void invertDWFDiracOperator(VectorTypeHost& psi, const VectorTypeHost& eta,
			      const double mass, const double M5, const int Ls,
			      const int kernelType,
			      const Complex boundaryConditions[4],
			      const int solverMethod, const int precondition,
			      const int maxIterations, const double tolerance,
			      const int verbosity, Complex* gaugeField,
			      const int L, const int T);
  }
}
#endif
