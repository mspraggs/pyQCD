#include <cuda_interface.h>

void invertDiracOperator(const VectorTypeHost& psi, const VectorTypeHost& eta,
			 LinearOperator* diracMatrix, const int solverMethod,
			 const int precondition, const int maxIterations,
			 const double tolerance, const int verbosity)
{
  VectorTypeDev psiDev = eta; // Put the source here temporarily

  if (solverMethod == 1) {
    // To save memory, we use the solution to hold the source while
    // we make it hermitian (to prevent data race)
    Complex* eta_ptr = thrust::raw_pointer_cast(&etaDev[0]);
    Complex* psi_ptr = thrust::raw_pointer_cast(&psiDev[0]);
    diracMatrix->makeHermitian(eta_ptr, psi_ptr);
  }
  else {
    etaDev = eta;
  }
  
  psiDev = VectorTypeDev(diracMatrix->num_rows, 0);

  cusp::default_monitor<Complex> monitor(etaDev, maxIterations, 0, tolerance);

  // Now do the inversion
  switch (solverMethod) {
  case 0:
    cusp::krylov::bicgstab(diracOperator, psiDev, etaDev, monitor);
    break;
  case 1:
    cusp::krylov::cg(diracOperator, psiDev, etaDev, monitor);
    break;
  case default:
    cusp::krylov::cg(diracOperator, psiDev, etaDev, monitor);
    break;    
  }
  if (verbosity > 0) {
    std::cout << "  -> Solver finished with residual of "
	      << monitor.residual_norm() << " in "
	      << monitor.iteration_count() << " iterations." << std::endl;
  }

  // Move the result to main memory
  psi = psiDev;
}
