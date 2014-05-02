#include <cuda_interface.h>

namespace pyQCD
{
  extern "C"
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

    CudaLinearOperator* diracOperator = new CudaWilson(mass, L, T, precondition,
					       hermitian, boundaryConditions,
					       gaugeField, true);

    if (verbosity > 0)
      std::cout << "  Done!" << std::endl;

    invertDiracOperator(psi, eta, diracOperator, solverMethod, precondition,
			maxIterations, tolerance, verbosity);

    delete diracOperator;
  }



  extern "C"
  void invertHamberWuDiracOperator(VectorTypeHost& psi,
				   const VectorTypeHost& eta,
				   const double mass,
				   const Complex boundaryConditions[4],
				   const int solverMethod,
				   const int precondition,
				   const int maxIterations,
				   const double tolerance,
				   const int verbosity, Complex* gaugeField,
				   const int L, const int T)
  {
    if (verbosity > 0)
      std::cout << "  Generating Dirac matrix..." << std::flush;

    bool hermitian = solverMethod == 1;

    CudaLinearOperator* diracOperator
      = new CudaHamberWu(mass, L, T, precondition,
			 hermitian, boundaryConditions,
			 gaugeField, true);

    if (verbosity > 0)
      std::cout << "  Done!" << std::endl;

    invertDiracOperator(psi, eta, diracOperator, solverMethod, precondition,
			maxIterations, tolerance, verbosity);

    delete diracOperator;
  }



  extern "C"
  void invertNaikDiracOperator(VectorTypeHost& psi, const VectorTypeHost& eta,
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

    CudaLinearOperator* diracOperator
      = new CudaNaik(mass, L, T, precondition,
		     hermitian, boundaryConditions,
		     gaugeField, true);
    
    if (verbosity > 0)
      std::cout << "  Done!" << std::endl;

    invertDiracOperator(psi, eta, diracOperator, solverMethod, precondition,
			maxIterations, tolerance, verbosity);

    delete diracOperator;
  }



  extern "C"
  void invertDWFDiracOperator(VectorTypeHost& psi, const VectorTypeHost& eta,
			      const double mass, const double M5, const int Ls,
			      const int kernelType,
			      const Complex boundaryConditions[4],
			      const int solverMethod, const int precondition,
			      const int maxIterations, const double tolerance,
			      const int verbosity, Complex* gaugeField,
			      const int L, const int T)
  {
    if (verbosity > 0)
      std::cout << "  Generating Dirac matrix..." << std::flush;

    bool hermitian = solverMethod == 1;

    CudaLinearOperator* diracOperator
      = new CudaDWF(mass, M5, Ls, kernelType, L, T, precondition,
		    hermitian, boundaryConditions, gaugeField);
    
    if (verbosity > 0)
      std::cout << "  Done!" << std::endl;

    invertDiracOperator(psi, eta, diracOperator, solverMethod, precondition,
			maxIterations, tolerance, verbosity);

    delete diracOperator;
  }



  void makeSource(VectorTypeDev& eta, const int site[4], const int spin,
		  const int colour, const CudaLinearOperator* smearingOperator)
  {
    cusp::blas::fill(eta, Complex(0.0, 0.0));
  
    int L = smearingOperator->L();
    int T = smearingOperator->T();

    int spatialIndex = site[3] + L * (site[2] + L * (site[1] + L * site[0]));
    int index = colour + 3 * (spin + 4 * spatialIndex);
    eta[index] = 1.0;

    VectorTypeDev chi = eta;
    (*smearingOperator)(eta, chi);
  }



  void invertDiracOperator(VectorTypeHost& psi, const VectorTypeHost& eta,
			   CudaLinearOperator* diracMatrix,
			   const int solverMethod,
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
      cusp::krylov::bicgstab(*diracMatrix, psiDev, etaDev, monitor);
      break;
    case 1:
      cusp::krylov::cg(*diracMatrix, psiDev, etaDev, monitor);
      break;
    default:
      cusp::krylov::cg(*diracMatrix, psiDev, etaDev, monitor);
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
			 const CudaLinearOperator* diracMatrix, const int site[4],
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

    CudaLinearOperator* sourceSmearingOperator
      = new CudaJacobiSmearing(nSourceSmears, sourceSmearingParameter,
			   diracMatrix->L(), diracMatrix->T(),
			   boundaryConditions, diracMatrix->links(), false);

    CudaLinearOperator* sinkSmearingOperator
      = new CudaJacobiSmearing(nSinkSmears, sinkSmearingParameter,
			   diracMatrix->L(), diracMatrix->T(),
			   boundaryConditions, diracMatrix->links(), false);

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
	  cusp::krylov::bicgstab(*diracMatrix, psi, eta, monitor);
	  break;
	case 1:
	  cusp::krylov::cg(*diracMatrix, psi, eta, monitor);
	  break;
	default:
	  cusp::krylov::cg(*diracMatrix, psi, eta, monitor);
	  break;    
	}

	if (verbosity > 0) {
	  std::cout << "  -> Solver finished with residual of "
		    << monitor.residual_norm() << " in "
		    << monitor.iteration_count() << " iterations." << std::endl;
	}
	// Copy the solution to the source before sink smearing
	cusp::copy(psi, eta);
	(*sinkSmearingOperator)(psi, eta);

	PropagatorTypeHost::column_view propagatorColumn
	  = result.column(3 * i + j);

	cusp::copy(psi, propagatorColumn);
      }
    }

    delete sourceSmearingOperator;
    delete sinkSmearingOperator;
  }
}
