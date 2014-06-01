#include <cuda_interface.h>

namespace pyQCD
{
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



  void diracOperatorFactory(CudaLinearOperator*& diracOperator,
			    const int action,
			    const int* intParams, const float* floatParams,
			    const Complex* complexParams,
			    const Complex* boundaryConditions,
			    const int L, const int T, const bool precondition,
			    const bool hermitian, Complex* links,
			    const bool copyLinks)
  {
    // Generates the specified Dirac operator with the specified paramaters

    switch (action) {
    case 0:
      diracOperator = new CudaWilson(floatParams[0], L, T, precondition,
				     hermitian, boundaryConditions, links,
				     copyLinks);
      break;
    case 1:
      diracOperator = new CudaHamberWu(floatParams[0], L, T, precondition,
				       hermitian, boundaryConditions, links,
				       copyLinks);
      break;
    case 2:
      diracOperator = new CudaNaik(floatParams[0], L, T, precondition,
				   hermitian, boundaryConditions, links,
				   copyLinks);
      break;
    case 3:
      diracOperator = new CudaDWF(floatParams[0], floatParams[1],
				  intParams[0], intParams[1], L, T,
				  precondition, hermitian, boundaryConditions,
				  links);
      break;
    }
  }


  extern "C"
  void invertDiracOperator(VectorTypeHost& psi, const int action,
			   const int* intParams, const float* floatParams,
			   const Complex* complexParams,
			   const Complex* boundaryConditions,
			   const VectorTypeHost& eta, const int solverMethod,
			   const int precondition, int* maxIterations,
			   double* tolerance, const int verbosity,
			   const int L, const int T, Complex* gaugeField)
  {
    int solveSize = (precondition > 0) ? psi.size() / 2 : psi.size();
    
    VectorTypeDev psiDev(solveSize, 0.0); // Put the source here temporarily
    VectorTypeDev etaDev(solveSize, 0.0);

    CudaLinearOperator* diracMatrix;
    diracOperatorFactory(diracMatrix, action, intParams, floatParams,
			 complexParams, boundaryConditions, L, T, precondition,
			 solverMethod == 1, gaugeField, true);

    VectorTypeHost etaEvenOdd;
    VectorTypeHost psiEvenOdd;

    if (precondition == 0) {
      etaDev = eta;
    }
    else {
      // Here we construct the odd source
      etaEvenOdd.resize(eta.size());
      psiEvenOdd.resize(psi.size());
      diracMatrix->makeEvenOdd(thrust::raw_pointer_cast(&etaEvenOdd[0]),
			       thrust::raw_pointer_cast(&eta[0]));

      VectorTypeHost::view etaEven(etaEvenOdd.begin(),
				   etaEvenOdd.begin() + solveSize);
      VectorTypeHost::view etaOdd(etaEvenOdd.begin() + solveSize,
				  etaEvenOdd.end());

      cusp::copy(etaOdd, etaDev); // etaD <- eta_o
      cusp::copy(etaEven, psiDev); // psi D <- eta_e

      // etaD <- etaD - Moe Mee^-1 psiD
      // eta_o <- eta_o - Moe Mee^-1 eta_e
      diracMatrix->makeEvenOddSource(thrust::raw_pointer_cast(&etaDev[0]),
				     thrust::raw_pointer_cast(&psiDev[0]),
				     thrust::raw_pointer_cast(&etaDev[0]));
    }

    if (solverMethod == 1) {
      psiDev = etaDev;
      // To save memory, we use the solution to hold the source while
      // we make it hermitian (to prevent data race)
      Complex* eta_ptr = thrust::raw_pointer_cast(&etaDev[0]);
      Complex* psi_ptr = thrust::raw_pointer_cast(&psiDev[0]);
      diracMatrix->makeHermitian(eta_ptr, psi_ptr);
      cusp::blas::fill(psiDev, Complex(0.0, 0.0));
    }

    cusp::default_monitor<Complex> monitor(etaDev, *maxIterations, 0, *tolerance);

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
    *tolerance = monitor.residual_norm();
    *maxIterations = monitor.iteration_count();
    if (verbosity > 0) {
      std::cout << "  -> Solver finished with residual of "
		<< monitor.residual_norm() << " in "
		<< monitor.iteration_count() << " iterations." << std::endl;
    }

    if (precondition == 0) {
      // Move the result to main memory
      psi = psiDev;
    }
    else {
      VectorTypeHost::view psiEven(psiEvenOdd.begin(),
				   psiEvenOdd.begin() + solveSize);
      VectorTypeHost::view psiOdd(psiEvenOdd.begin() + solveSize,
				  psiEvenOdd.end());
      VectorTypeHost::view etaEven(etaEvenOdd.begin(),
				   etaEvenOdd.begin() + solveSize);
      cusp::copy(psiDev, psiOdd); // psi_o <- psiD
      cusp::copy(etaEven, etaDev); // etaD <- eta_e

      diracMatrix->applyEvenEvenInv(thrust::raw_pointer_cast(&etaDev[0]),
				    thrust::raw_pointer_cast(&etaDev[0]));

      diracMatrix->makeEvenOddSolution(thrust::raw_pointer_cast(&psiDev[0]),
				       thrust::raw_pointer_cast(&etaDev[0]),
				       thrust::raw_pointer_cast(&psiDev[0]));

      cusp::copy(psiDev, psiEven);

      diracMatrix->removeEvenOdd(thrust::raw_pointer_cast(&psi[0]),
				 thrust::raw_pointer_cast(&psiEvenOdd[0]));
    }

    delete diracMatrix;
  }


  
  extern "C"
  void computePropagator(PropagatorTypeHost& result,
			 const int action, const int* intParams,
			 const float* floatParams, const Complex* complexParams,
			 const Complex* boundaryConditions, const int site[4],
			 const int sourceSmearingType, const int nSourceSmears,
			 const float sourceSmearingParameter,
			 const int sinkSmearingType, const int nSinkSmears,
			 const float sinkSmearingParameter,
			 const int solverMethod, const int maxIterations,
			 const float tolerance, const int precondition,
			 const int verbosity, const int L, const int T,
			 Complex* gaugeField)
  {
    Complex tempBoundaryConditions[4] = {Complex(-1.0, 0.0),
					 Complex(1.0, 0.0),
					 Complex(1.0, 0.0),
					 Complex(1.0, 0.0)};

    CudaLinearOperator* diracMatrix;
    diracOperatorFactory(diracMatrix, action, intParams, floatParams,
			 complexParams, boundaryConditions, L, T, precondition,
			 solverMethod == 1, gaugeField, true);

    CudaLinearOperator* sourceSmearingOperator
      = new CudaJacobiSmearing(nSourceSmears, sourceSmearingParameter,
			       diracMatrix->L(), diracMatrix->T(),
			       tempBoundaryConditions, diracMatrix->links(),
			       false);

    CudaLinearOperator* sinkSmearingOperator
      = new CudaJacobiSmearing(nSinkSmears, sinkSmearingParameter,
			       diracMatrix->L(), diracMatrix->T(),
			       tempBoundaryConditions, diracMatrix->links(),
			       false);

    int solveSize = diracMatrix->num_rows;
    int N = (precondition > 0) ? 2 * solveSize : solveSize;

    VectorTypeDev eta(N, Complex(0.0, 0.0));
    VectorTypeDev psi(N, Complex(0.0, 0.0));

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 3; ++j) {
	if (verbosity > 0)
	  std::cout << "  Inverting for spin " << i
		    << " and colour " << j << "..." << std::endl;
      
	makeSource(eta, site, i, j, sourceSmearingOperator);

	if (precondition > 0) {
	  // This is ugly, there's so much transfer between host and device...
	  VectorTypeHost etaHost = eta;
	  VectorTypeHost etaEvenOdd(N, 0.0);

	  diracMatrix->makeEvenOdd(thrust::raw_pointer_cast(&etaEvenOdd[0]),
				   thrust::raw_pointer_cast(&etaHost[0]));

	  VectorTypeHost::view etaEvenHost(etaEvenOdd.begin(),
					   etaEvenOdd.begin() + solveSize);
	  VectorTypeHost::view etaOddHost(etaEvenOdd.begin() + solveSize,
					  etaEvenOdd.end());

	  VectorTypeDev::view etaEvenDev(eta.begin(),
					 eta.begin() + solveSize);
	  VectorTypeDev::view etaOddDev(eta.begin() + solveSize,
					eta.end());

	  cusp::copy(etaOddHost, etaOddDev);
	  cusp::copy(etaEvenHost, etaEvenDev);

	  
	  diracMatrix
	    ->makeEvenOddSource(thrust::raw_pointer_cast(&etaOddDev[0]),
				thrust::raw_pointer_cast(&etaEvenDev[0]),
				thrust::raw_pointer_cast(&etaOddDev[0]));
	}

	VectorTypeDev::view etaSolve((precondition > 0)
				     ? eta.begin() + solveSize
				     : eta.begin(),
				     eta.end());

	VectorTypeDev::view psiSolve((precondition > 0)
				     ? psi.begin() + solveSize
				     : psi.begin(),
				     psi.end());
	
	if (solverMethod == 1) {
	  // To save memory, we use the solution to hold the source while
	  // we make it hermitian (to prevent data race)
	  cusp::copy(etaSolve, psiSolve);
	  Complex* eta_ptr = thrust::raw_pointer_cast(&etaSolve[0]);
	  Complex* psi_ptr = thrust::raw_pointer_cast(&psiSolve[0]);
	  diracMatrix->makeHermitian(eta_ptr, psi_ptr);
	  cusp::blas::fill(psiSolve, Complex(0.0, 0.0));
	}
      
	cusp::default_monitor<Complex> monitor(etaSolve, maxIterations, 0, tolerance);
  
	// Now do the inversion
	switch (solverMethod) {
	case 0:
	  cusp::krylov::bicgstab(*diracMatrix, psiSolve, etaSolve, monitor);
	  break;
	case 1:
	  cusp::krylov::cg(*diracMatrix, psiSolve, etaSolve, monitor);
	  break;
	default:
	  cusp::krylov::cg(*diracMatrix, psiSolve, etaSolve, monitor);
	  break;    
	}

	if (verbosity > 0) {
	  std::cout << "  -> Solver finished with residual of "
		    << monitor.residual_norm() << " in "
		    << monitor.iteration_count() << " iterations." << std::endl;
	}

	if (precondition > 0) {
	  // Again, there's so much back and forth between the host and device here
	  // that memory latency will be huge.
	  VectorTypeHost psiLexico(N, 0.0);
	  VectorTypeHost psiEvenOdd(N, 0.0);

	  VectorTypeDev::view psiEvenDev(psi.begin(),
					 psi.begin() + solveSize);
	  VectorTypeDev::view psiOddDev(psi.begin() + solveSize,
					psi.end());

	  VectorTypeHost::view psiEvenHost(psiEvenOdd.begin(),
					   psiEvenOdd.begin() + solveSize);
	  VectorTypeHost::view psiOddHost(psiEvenOdd.begin() + solveSize,
					  psiEvenOdd.end());

	  cusp::copy(psiOddDev, psiOddHost);

	  diracMatrix->applyEvenEvenInv(thrust::raw_pointer_cast(&psi[0]),
					thrust::raw_pointer_cast(&eta[0]));

	  diracMatrix
	    ->makeEvenOddSolution(thrust::raw_pointer_cast(&psiOddDev[0]),
				  thrust::raw_pointer_cast(&psiEvenDev[0]),
				  thrust::raw_pointer_cast(&psiOddDev[0]));

	  cusp::copy(psiOddDev, psiEvenHost);

	  diracMatrix
	    ->removeEvenOdd(thrust::raw_pointer_cast(&psiLexico[0]),
			    thrust::raw_pointer_cast(&psiEvenOdd[0]));
	
	  eta = psiLexico;
	}
	else
	  eta = psi;
	
	(*sinkSmearingOperator)(eta, psi);

	PropagatorTypeHost::column_view propagatorColumn
	  = result.column(3 * i + j);
	cusp::copy(psi, propagatorColumn);
      }
    }

    delete sourceSmearingOperator;
    delete sinkSmearingOperator;
    delete diracMatrix;
  }
}
