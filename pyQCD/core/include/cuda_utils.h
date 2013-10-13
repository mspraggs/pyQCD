#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <vector>

#include <cusp/ell_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/cg.h>
#include <cusp/complex.h>
#include <cusp/print.h>
#include <cusp/copy.h>
#include <cusp/blas.h>

#include <iostream>

typedef cusp::device_memory devMem;
typedef cusp::host_memory hostMem;
typedef cusp::csr_matrix<int, cusp::complex<float>, hostMem> complexHybridHost;
typedef cusp::ell_matrix<int, cusp::complex<float>, devMem> complexHybridDev;

namespace pyQCD
{
  namespace cuda {
    void createSource(const int site[4], const int spin, const int colour,
		      const complexHybridDev& smearingOperator,
		      cusp::array1d<cusp::complex<float>, devMem>& source,
		      cusp::array1d<cusp::complex<float>, devMem>&
		      tempSource);

    void bicgstab(const complexHybridHost& hostDirac,
		  const complexHybridHost& hostSourceSmear,
		  const complexHybridHost& hostSinkSmear,
		  const int spatialIndex,
		  cusp::array2d<cusp::complex<float>, hostMem>& propagator,
		  const int verbosity);

    void cg(const complexHybridHost& hostDiracDiracAdjoint,
	    const complexHybridHost& hostDiracAdjoint,
	    const complexHybridHost& hostSourceSmear,
	    const complexHybridHost& hostSinkSmear,
	    const int spatialIndex,
	    cusp::array2d<cusp::complex<float>, hostMem>& propagator,
	    const int verbosity);
  }
}
#endif


