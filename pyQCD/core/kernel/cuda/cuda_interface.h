#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H

#include <cusp/linear_operator.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/bicgstab.h>

#include <utils.h>
#include <wilson.h>
#include <hamber_wu.h>
#include <naik.h>
#include <dwf.h>

void makeSource(VectorTypeDev& eta, const int site[4], const int spin,
		const int colour, const LinearOperator* smearingOperator);

void invertDiracOperator(const VectorTypeHost& psi, const VectorTypeHost& eta,
			 LinearOperator* diracMatrix, const int solverMethod,
			 const int precondition, const int maxIterations,
			 const double tolerance, const int verbosity);

void computePropagator(PropagatorTypeHost& result,
		       const LinearOperator* diracMatrix, const int site[4],
		       const int sourceSmearingType, const int nSourceSmears,
		       const float sourceSmearingParameter,
		       const int sinkSmearingType, const int nSinkSmears,
		       const float sinkSmearingParameter,
		       const int solverMethod, const int maxIterations,
		       const float tolerance, const int precondition,
		       const int verbosity);

#endif
