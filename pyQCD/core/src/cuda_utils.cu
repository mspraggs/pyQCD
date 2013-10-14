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
      cusp::blas::fill(tempSource, cusp::complex<float>(0.0, 0.0));
      tempSource[index] = cusp::complex<float>(1.0, 0.0);

      cusp::multiply(smearingOperator, tempSource, source);
    }

  
  
    void bicgstab(const complexHybridHost& hostDirac,
		  const complexHybridHost& hostSourceSmear,
		  const complexHybridHost& hostSinkSmear,
		  const int spatialIndex,
		  cusp::array2d<cusp::complex<float>, hostMem>& propagator,
		  const int verbosity)
    {
      // Get the size of the Dirac matrix
      int nCols = hostDirac.num_cols;
      // Transfer the Dirac and smearing matrices to the device.
      if (verbosity > 0)
	std::cout << "  Transferring matrices to device..." << std::flush;
      complexHybridDev devDirac = hostDirac;
      complexHybridDev devSourceSmear = hostSourceSmear;
      complexHybridDev devSinkSmear = hostSinkSmear;
      if (verbosity > 0)
	std::cout << " Done!" << std::endl;

      // Create some device arrays to hold the source and solution
      cusp::array1d<cusp::complex<float>, devMem>
	source(nCols, cusp::complex<float>(0, 0));
      cusp::array1d<cusp::complex<float>, devMem>
	solution(nCols, cusp::complex<float>(0, 0));

      // And another load as we'll need duplicates for the
      // multiplication routines
      cusp::array1d<cusp::complex<float>, devMem>
	tempSource(nCols, cusp::complex<float>(0, 0));
      cusp::array1d<cusp::complex<float>, devMem>
	tempSolution(nCols, cusp::complex<float>(0, 0));

      // Create a temporary propagator variable for fast access
      cusp::array2d<cusp::complex<float>, devMem>
	tempPropagator(nCols, 12, cusp::complex<float>(0, 0));

      // Create the preconditioner
      cusp::precond::diagonal<cusp::complex<float>, devMem>
	preconditioner(devDirac);
      
      // Loop through all spins and colours and do the inversions
      for (int i = 0; i < 4; ++i) {
	for (int j = 0; j < 3; ++j) {
	  if (verbosity > 0)
	    std::cout << "  Inverting for spin " << i
		      << " and colour " << j << "..." << std::endl;
	  // Create the source using the smearing operator
	  createSource(spatialIndex, i, j, devSourceSmear, source,
		       tempSource);
	  // Set up the monitor for use in the solver
	  cusp::default_monitor<cusp::complex<float> >
	    monitor(source, 1000, 0, 1e-8);
	  
	  // Do the inversion
	  cusp::krylov::bicgstab(devDirac, solution, source, monitor,
				 preconditioner);
	  // Smear at the sink
	  cusp::multiply(devSinkSmear, solution, tempSolution);
	  // Create a view to the relevant column of the propagator
	  cusp::array2d<cusp::complex<float>, devMem>::column_view
	    propagatorView = tempPropagator.column(j + 3 * i);
	  // Copy the solution to the propagator output
	  cusp::copy(tempSolution, propagatorView);
	  if (verbosity > 0)
	    std::cout << "  -> Inversion reached tolerance of " 
		      << monitor.residual_norm() << " in "
		      << monitor.iteration_count() << " iterations."
		      << std::endl;
	}
      }
      // Move the propagator back into main memory
      if (verbosity > 0)
	std::cout << "  Transferring propagators to main memory..."
		  << std::flush;
      propagator = tempPropagator;
      if (verbosity > 0)
	std::cout << " Done!" << std::endl;
    }

  
  
    void cg(const complexHybridHost& hostDiracDiracAdjoint,
	    const complexHybridHost& hostDiracAdjoint,
	    const complexHybridHost& hostSourceSmear,
	    const complexHybridHost& hostSinkSmear,
	    const int spatialIndex,
	    cusp::array2d<cusp::complex<float>, hostMem>& propagator,
	    const int verbosity)
    {
      // Get the size of the Dirac matrix
      int nCols = hostDiracDiracAdjoint.num_cols;
      // Transfer the Dirac and smearing matrices to the device.
      if (verbosity > 0)
	std::cout << "  Transferring matrices to device..." << std::flush;
      complexHybridDev devM = hostDiracDiracAdjoint;
      complexHybridDev devDadj = hostDiracAdjoint;
      complexHybridDev devSourceSmear = hostSourceSmear;
      complexHybridDev devSinkSmear = hostSinkSmear;
      if (verbosity > 0)
	std::cout << " Done!" << std::endl;

      // Create some device arrays to hold the source and solution
      cusp::array1d<cusp::complex<float>, devMem>
	source(nCols, cusp::complex<float>(0, 0));
      cusp::array1d<cusp::complex<float>, devMem>
	solution(nCols, cusp::complex<float>(0, 0));

      // And another load as we'll need duplicates for the
      // multiplication routines
      cusp::array1d<cusp::complex<float>, devMem>
	tempSource(nCols, cusp::complex<float>(0, 0));
      cusp::array1d<cusp::complex<float>, devMem>
	tempSolution(nCols, cusp::complex<float>(0, 0));

      // Create a temporary propagator variable for fast access
      cusp::array2d<cusp::complex<float>, devMem>
	tempPropagator(nCols, 12, cusp::complex<float>(0, 0));

      // Create the preconditioner
      cusp::precond::diagonal<cusp::complex<float>, devMem>
	preconditioner(devM);

      // Loop through all spins and colours and do the inversions
      for (int i = 0; i < 4; ++i) {
	for (int j = 0; j < 3; ++j) {
	  if (verbosity > 0)
	    std::cout << "  Inverting for spin " << i
		      << " and colour " << j << "..." << std::endl;
	  // Create the source using the smearing operator
	  createSource(spatialIndex, i, j, devSourceSmear, source,
		       tempSource);
	  // Set up the monitor for use in the solver
	  cusp::default_monitor<cusp::complex<float> >
	    monitor(source, 1000, 0, 1e-8);
	  
	  // Do the inversion
	  cusp::krylov::cg(devM, solution, source, monitor,
			   preconditioner);
	  // Smear at the sink
	  cusp::multiply(devDadj, solution, tempSolution);
	  
	  cusp::multiply(devSinkSmear, tempSolution, solution);
	  // Create a view to the relevant column of the propagator
	  cusp::array2d<cusp::complex<float>, devMem>::column_view
	    propagatorView = tempPropagator.column(j + 3 * i);
	  // Copy the solution to the propagator output
	  cusp::copy(solution, propagatorView);
	  if (verbosity > 0)
	    std::cout << "  -> Inversion reached tolerance of " 
		      << monitor.residual_norm() << " in "
		      << monitor.iteration_count() << " iterations."
		      << std::endl;
	}
      }
      // Move the propagator back into main memory
      if (verbosity > 0)
	std::cout << "  Transferring propagators to main memory..."
		  << std::flush;
      propagator = tempPropagator;
      if (verbosity > 0)
	std::cout << " Done!" << std::endl;
    }
  }
}
