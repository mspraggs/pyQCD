#include <cuda_utils.h>

namespace pyQCD
{
  namespace cuda
  {
    void createSource(const int spatialIndex, const int spin, const int colour,
		      const cusp::hyb_matrix<int, cusp::complex<double>,
					     devMem> smearingOperator,
		      cusp::array1d<cusp::complex<double>, devMem>& source)
    {
      int index = colour + 3 * (spin + spatialIndex);

      for (int i = 0; i < source.size(); ++i)
	source[i] = cusp::complex<double>(0.0, 0.0);

      source[index] = cusp::complex<double>(1.0, 0.0);

      cusp::multiply(smearingOperator, source, source);
    }

  
  
    void bicgstab(const complexHybridHost hostDirac,
		  const complexHybridHost hostSourceSmear,
		  const complexHybridHost hostSinkSmear,
		  const int spatialIndex,
		  cusp::array2d<cusp::complex<double>, hostMem> propagator)
    {
      int nCols = hostDirac.num_cols;
      std::cout << nCols << std::endl;
    
      complexHybridDev devDirac = hostDirac;
      complexHybridDev devSourceSmear = hostSourceSmear;
      complexHybridDev devSinkSmear = hostSinkSmear;

      cusp::array1d<cusp::complex<double>,
		    devMem> source(nCols, cusp::complex<double>(0, 0));
      cusp::array1d<cusp::complex<double>,
		    devMem> solution(nCols, cusp::complex<double>(0, 0));

      for (int i = 0; i < 4; ++i) {
	for (int j = 0; j < 3; ++j) {
	  createSource(spatialIndex, i, j, devSourceSmear, source);

	  cusp::verbose_monitor<cusp::complex<double> > monitor(source, 100,
							    1e-3);
	  cusp::identity_operator<cusp::complex<double>, devMem>
	    preconditioner(devDirac.num_rows, devDirac.num_rows);

	  cusp::krylov::bicgstab(devDirac, solution, source, monitor,
				 preconditioner);
       
	  cusp::multiply(devDirac, solution, solution);

	  for (int k = 0; k < nCols; ++k)
	    propagator(k, j + 3 * i) = solution[k];
	}
      }
    }
  }
}
