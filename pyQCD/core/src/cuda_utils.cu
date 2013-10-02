#include <cuda_utils.h>

namespace pyQCD
{
  namespace cuda
  {
    void createSource(const int spatialIndex, const int spin, const int colour,
		      const complexHybridDev& smearingOperator,
		      cusp::array1d<cusp::complex<float>, devMem>& source,
		      cusp::array1d<cusp::complex<float>, devMem>&
		      tempSource)
    {
      int index = colour + 3 * (spin + spatialIndex);
      
      tempSource[index] = cusp::complex<float>(1.0, 0.0);

      cusp::multiply(smearingOperator, tempSource, source);
    }

  
  
    void bicgstab(const complexHybridHost& hostDirac,
		  const complexHybridHost& hostSourceSmear,
		  const complexHybridHost& hostSinkSmear,
		  const int spatialIndex,
		  cusp::array2d<cusp::complex<float>, hostMem>& propagator)
    {
      int nCols = hostDirac.num_cols;
      
      complexHybridDev devDirac = hostDirac;
      complexHybridDev devSourceSmear = hostSourceSmear;
      complexHybridDev devSinkSmear = hostSinkSmear;

      cusp::array1d<cusp::complex<float>,
		    devMem> source(nCols, cusp::complex<float>(0, 0));
      cusp::array1d<cusp::complex<float>,
		    devMem> solution(nCols, cusp::complex<float>(0, 0));

      cusp::array1d<cusp::complex<float>,
		    devMem> tempSource(nCols, cusp::complex<float>(0, 0));
      cusp::array1d<cusp::complex<float>, devMem>
	tempSolution(nCols, cusp::complex<float>(0, 0));

      cusp::array1d<cusp::complex<float>, devMem>::view
	solutionView = cusp::make_array1d_view(tempSolution);

      cusp::array2d<cusp::complex<float>, devMem>
	tempPropagator(nCols, 12, cusp::complex<float>(0, 0));

      for (int i = 0; i < 4; ++i) {
	for (int j = 0; j < 3; ++j) {
	  createSource(spatialIndex, i, j, devSourceSmear, source,
		       tempSource);
	  tempSource[j + 3 * (i + spatialIndex)]
	    = cusp::complex<float>(0, 0);

	  cusp::verbose_monitor<cusp::complex<float> > monitor(source, 100,
							       1e-3);
	  cusp::identity_operator<cusp::complex<float>, devMem>
	    preconditioner(devDirac.num_rows, devDirac.num_rows);

	  cusp::krylov::bicgstab(devDirac, solution, source, monitor,
				 preconditioner);
       
	  cusp::multiply(devSinkSmear, solution, tempSolution);

	  cusp::array2d<cusp::complex<float>, devMem>::column_view
	    propagatorView = tempPropagator.column(j + 3 * i);
	  cusp::copy(tempSolution, propagatorView);// = tempSolution;
	  //cusp::convert(solutionView, tempPropagator.column(j + 3 * i));
	  //tempPropagator.column(j + 3 * i) = solutionView;
	  //for (int k = 0; k < nCols; ++k) {
	  //  tempPropagator(k, j + 3 * i) = tempSolution[k];
	  //}
	}
      }
      propagator = tempPropagator;
    }
  }
}
