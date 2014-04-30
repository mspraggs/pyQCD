#include <cuda_interface.h>

void makeSource(VectorTypeDev& eta, const int site[4], const int spin,
		const int colour, const LinearOperator* smearingOperator)
{
  cusp::blas::fill(eta, Complex(0.0, 0.0));
  
  int L = smearingOperator->L();
  int T = smearingOperator->T();

  int N = 12 * L * L * L * T;
  int spatialIndex = site[3] + L * (site[2] + L * (site[1] + L * site[0]));
  int index = colour + 3 * (spin + 4 * spatialIndex);
  eta[index] = 1.0;

  VectorTypeDev chi = eta;
  (*smearingOperator)(eta, chi);
}



void invertWilsonDiracOperator(VectorTypeHost& psi, const VectorTypeHost& eta,
			       const double mass,
			       const Complex boundaryConditions[4],
			       const int solverMethod, const int precondition,
			       const int maxIterations, const double tolerance,
			       const int verbosity, Complex* gaugeField,
			       const int L, const int T)
{
  if (verbosity > 0)
    std::cout << "  Generating Dirac matrix..." << std::flush;

  bool hermitian = solverMethod == 1;

  LinearOperator* diracOperator = new Wilson(mass, L, T, preconditionB, hermitian,
					     boundaryConditions, gaugeField,
					     true);

  if (verbosity > 0)
    std::cout << " Done!" << std::endl;

  invertDiracOperator(psi, eta, diracOperator, solverMethod, precondition,
		      maxIterations, tolerance, verbosity);

  delete diracOperator;
}



void invertDiracOperator(VectorTypeHost& psi, const VectorTypeHost& eta,
			 LinearOperator* diracMatrix, const int solverMethod,
			 const int precondition, const int maxIterations,
			 const double tolerance, const int verbosity)
{
  VectorTypeDev psiDev = eta; // Put the source here temporarily
  VectorTypeDev etaDev = eta;

  if (solverMethod == 1) {
    // To save memory, we use the solution to hold the source while
    // we make it hermitian (to prevent data race)
    Complex* eta_ptr = thrust::raw_pointer_cast(&etaDev[0]);
    Complex* psi_ptr = thrust::raw_pointer_cast(&psiDev[0]);
    diracMatrix->makeHermitian(eta_ptr, psi_ptr);
    cusp::blas::fill(psiDev, Complex(0.0, 0.0));
  }

  cusp::default_monitor<Complex> monitor(etaDev, maxIterations, 0, tolerance);

  // Now do the inversion
  switch (solverMethod) {
  case 0:
    cusp::krylov::bicgstab(diracMatrix, psiDev, etaDev, monitor);
    break;
  case 1:
    cusp::krylov::cg(diracMatrix, psiDev, etaDev, monitor);
    break;
  default:
    cusp::krylov::cg(diracMatrix, psiDev, etaDev, monitor);
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



void computePropagator(PropagatorTypeHost& result,
		       const LinearOperator* diracMatrix, const int site[4],
		       const int sourceSmearingType, const int nSourceSmears,
		       const float sourceSmearingParameter,
		       const int sinkSmearingType, const int nSinkSmears,
		       const float sinkSmearingParameter,
		       const int solverMethod, const int maxIterations,
		       const float tolerance, const int precondition,
		       const int verbosity)
{
  Complex boundaryConditions[4] = {Complex(-1.0, 0.0),
				   Complex(1.0, 0.0),
				   Complex(1.0, 0.0),
				   Complex(1.0, 0.0)};

  LinearOperator* sourceSmearingOperator
    = new JacobiSmearing(nSourceSmears, sourceSmearingParameter,
			 diracMatrix->L(), diracMatrix->T(), boundaryConditions,
			 diracMatrix->links(), false);

  LinearOperator* sinkSmearingOperator
    = new JacobiSmearing(nSinkSmears, sinkSmearingParameter,
			 diracMatrix->L(), diracMatrix->T(), boundaryConditions,
			 diracMatrix->links(), false);

  VectorTypeDev eta(diracMatrix->num_rows, Complex(0.0, 0.0));
  VectorTypeDev psi(diracMatrix->num_rows, Complex(0.0, 0.0));

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (verbosity > 0)
	std::cout << "  Inverting for spin " << i
		  << " and colour " << j << "..." << std::flush;
      
      makeSource(eta, site, i, j, sourceSmearingOperator);

      if (solverMethod == 1) {
	// To save memory, we use the solution to hold the source while
	// we make it hermitian (to prevent data race)
	cusp::copy(eta, psi);
	Complex* eta_ptr = thrust::raw_pointer_cast(&eta[0]);
	Complex* psi_ptr = thrust::raw_pointer_cast(&psi[0]);
	diracMatrix->makeHermitian(eta_ptr, psi_ptr);
	cusp::blas::fill(psi, Complex(0.0, 0.0));
      }
      
      cusp::default_monitor<Complex> monitor(eta, maxIterations, 0, tolerance);
  
      // Now do the inversion
      switch (solverMethod) {
      case 0:
	cusp::krylov::bicgstab(diracMatrix, psi, eta, monitor);
	break;
      case 1:
	cusp::krylov::cg(diracMatrix, psi, eta, monitor);
	break;
      case default:
	cusp::krylov::cg(diracMatrix, psi, eta, monitor);
	break;    
      }

      if (verbosity > 0) {
	std::cout << "  -> Solver finished with residual of "
		  << monitor.residual_norm() << " in "
		  << monitor.iteration_count() << " iterations." << std::endl;
      }
      
      cusp::copy(psi, eta); // Copy the solution to the source before sink smearing
      (*sinkSmearingOperator)(psi, eta);

      VectorTypeHost::column_view propagatorColumn
	= propagator.column(3 * i + j);

      cusp::copy(psi, propagatorColumn);
    }
  }

  delete sourceSmearingOperator;
  delete sinkSmearingOperator;
}
