#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <vector>

#include <cusp/hyb_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/complex.h>
#include <cusp/print.h>

#include <iostream>

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>

typedef cusp::device_memory devMem;
typedef cusp::host_memory hostMem;
typedef cusp::coo_matrix<int, cusp::complex<double>, hostMem> complexHybridHost;
typedef cusp::hyb_matrix<int, cusp::complex<double>, devMem> complexHybridDev;

namespace pyQCD
{
  namespace cuda {
    void createSource(const int site[4], const int spin, const int colour,
		      const cusp::hyb_matrix<int, cusp::complex<double>,
		      devMem>& smearingOperator,
		      cusp::array1d<cusp::complex<double>, devMem> source);

    void bicgstab(const complexHybridHost hostDirac,
		  const complexHybridHost hostSourceSmear,
		  const complexHybridHost hostSinkSmear,
		  const int spatialIndex,
		  cusp::array2d<cusp::complex<double>, hostMem> propagator);

    void cudaCG();
  }
}
#endif


