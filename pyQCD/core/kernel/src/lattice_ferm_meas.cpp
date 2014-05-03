#include <lattice.hpp>
#include <utils.hpp>
#include <linear_operators.hpp>

VectorXcd
Lattice::makeSource(const int site[4], const int spin, const int colour,
		    LinearOperator* smearingOperator)
{
  // Generate a (possibly smeared) quark source at the given site, spin and
  // colour
  int nIndices = 3 * this->nLinks_;
  VectorXcd source(nIndices);
  source.setZero(nIndices);

  // Index for the vector point source
  int spatial_index = pyQCD::getLinkIndex(site[0], site[1], site[2], site[3], 0,
					  this->spatialExtent);
	
  // Set the point source
  int index = colour + 3 * (spin + spatial_index);
  source(index) = 1.0;

  // Now apply the smearing operator
  source = smearingOperator->apply(source);

  return source;
}



void Lattice::diracOperatorFactory(
  LinearOperator*& diracOperator, const int action, const vector<int>& intParams,
  const vector<double>& floatParams,
  const vector<complex<double> >& complexParams,
  const vector<complex<double> >& boundaryConditions)
{
  // Generates the specified Dirac operator with the specified parameters

  switch (action) {
  case pyQCD::wilson:
    diracOperator = new Wilson(floatParams[0], boundaryConditions, this);
    break;
  case pyQCD::hamberWu:
    diracOperator = new HamberWu(floatParams[0], boundaryConditions, this);
    break;
  case pyQCD::naik:
    diracOperator = new Naik(floatParams[0], boundaryConditions, this);
    break;
  case pyQCD::dwf:
    diracOperator = new DWF(floatParams[0], floatParams[1], intParams[0],
			    intParams[1], boundaryConditions, this);
    break;
  }
}



vector<MatrixXcd>
Lattice::computePropagator(const int action, const vector<int>& intParams,
			   const vector<double>& floatParams,
			   const vector<complex<double> >& complexParams,
			   const vector<complex<double> >& boundaryConditions,
			   int site[4], const int nSmears,
			   const double smearingParameter,
			   const int sourceSmearingType,
			   const int nSourceSmears,
			   const double sourceSmearingParameter,
			   const int sinkSmearingType,
			   const int nSinkSmears,
			   const double sinkSmearingParameter,
			   const int solverMethod,
			   const int maxIterations,
			   const double tolerance,
			   const int precondition,
			   const int verbosity)
{
  // Computes the propagator vectors for the 12 spin-colour indices at
  // the given lattice site, using the Dirac operator
  // First off, save the current links and smear all time slices

  GaugeField templinks;
  if (nSmears > 0) {
    templinks = this->links_;
    for (int time = 0; time < this->temporalExtent; time++) {
      this->smearLinks(time, nSmears, smearingParameter);
    }
  }

  // How many indices are we dealing with?
  int nSites = this->nLinks_ / 4;

  // Declare a variable to hold our propagator
  vector<MatrixXcd> propagator(nSites, MatrixXcd::Zero(12, 12));

#ifdef USE_CUDA

  Complex* gaugeField = new Complex[9 * this->nLinks_];
  pyQCD::eigenToCusp(gaugeField, this->links_);

  int* cuspIntParams = new int[intParams.size()];
  float* cuspFloatParams = new float[floatParams.size()];
  Complex* cuspComplexParams = new Complex[complexParams.size()];
  Complex cuspBoundaryConditions[4];
  
  pyQCD::convertLinopParams(cuspIntParams, intParams,
			    cuspFloatParams, floatParams,
			    cuspComplexParams, complexParams);

  pyQCD::eigenToCusp(cuspBoundaryConditions, boundaryConditions);

  PropagatorTypeHost cuspProp(12 * nSites, 12, 0.0);

  pyQCD::computePropagator(cuspProp, action, cuspIntParams, cuspFloatParams,
			   cuspComplexParams, cuspBoundaryConditions,
			   site, sourceSmearingType, nSourceSmears,
			   sourceSmearingParameter, sinkSmearingType,
			   nSinkSmears, sinkSmearingParameter, solverMethod,
			   maxIterations, tolerance, precondition, verbosity,
			   this->spatialExtent, this->temporalExtent,
			   gaugeField);

  delete[] cuspIntParams;
  delete[] cuspFloatParams;
  delete[] cuspComplexParams;
  delete[] gaugeField;

  propagator = pyQCD::cuspToEigen(cuspProp);

#else
  
  vector<complex<double> >
    smearingBoundaryConditions(4, complex<double>(1.0, 0.0));
  smearingBoundaryConditions[0] = complex<double>(-1.0, 0.0);

  // Create the source and sink smearing operators
  // TODO: If statements for different types of smearing/sources
  LinearOperator* sourceSmearingOperator 
    = new JacobiSmearing(nSourceSmears, sourceSmearingParameter,
			 smearingBoundaryConditions, this);
  LinearOperator* sinkSmearingOperator 
    = new JacobiSmearing(nSinkSmears, sinkSmearingParameter,
			 smearingBoundaryConditions, this);

  LinearOperator* diracMatrix;
  this->diracOperatorFactory(diracMatrix, action, intParams, floatParams,
			     complexParams, boundaryConditions);

  // Loop through colour and spin indices and invert propagator
  for (int i = 0; i < 4; ++i) {
    for(int j = 0; j < 3; ++j) {
      if (verbosity > 0)
	cout << "  Inverting for spin " << i
	     << " and colour " << j << "..." << flush;
      // Create the source vector
      VectorXcd source = this->makeSource(site, i, j, sourceSmearingOperator);

      // Do the inversion
      double residual = tolerance;
      int iterations = maxIterations;
      double time = 0.0;
      
      VectorXcd solution(3 * this->nLinks_);

      switch (solverMethod) {
      case pyQCD::bicgstab:
	solution = bicgstab(diracMatrix, source, residual, iterations, time,
			    precondition);
	break;
      case pyQCD::cg:
	solution = cg(diracMatrix, source, residual, iterations, time,
		      precondition);
	break;
      case pyQCD::gmres:
	solution = gmres(diracMatrix, source, residual, iterations, time,
			 precondition);
	break;
      default:
	solution = cg(diracMatrix, source, residual, iterations, time,
		      precondition);
	break;	
      }

      // Smear the sink
      solution = sinkSmearingOperator->apply(solution);
      
      // Add the result to the propagator matrix
      for (int k = 0; k < nSites; ++k) {
	for (int l = 0; l < 12; ++l) {
	  propagator[k](l, j + 3 * i) = solution(12 * k + l);
	}
      }
      if (verbosity > 0) {
	cout << " Done!" << endl;
	cout << "  -> Solver finished with residual of "
	     << residual << " in " << iterations << " iterations." << endl;
	cout << "  -> CPU time: " << time << " seconds" << endl;
      }
    }
  }

  delete sourceSmearingOperator;
  delete sinkSmearingOperator;
  delete diracMatrix;

#endif

  // Restore the non-smeared gauge field
  if (nSmears > 0)
    this->links_ = templinks;

  return propagator;
}



VectorXcd Lattice::invertDiracOperator(
  const int action, const vector<int>& intParams,
  const vector<double>& floatParams,
  const vector<complex<double> >& complexParams,
  const vector<complex<double> >& boundaryConditions, const VectorXcd& eta,
  const int solverMethod, const int precondition, const int maxIterations,
  const double tolerance, const int verbosity)
{
  // Inverts the supplied Dirac operator on the supplied source.

  VectorXcd psi = VectorXcd::Zero(eta.size());

#ifdef USE_CUDA

  Complex* gaugeField = new Complex[9 * this->nLinks_];
  pyQCD::eigenToCusp(gaugeField, this->links_);

  int* cuspIntParams = new int[intParams.size()];
  float* cuspFloatParams = new float[floatParams.size()];
  Complex* cuspComplexParams = new Complex[complexParams.size()];
  Complex cuspBoundaryConditions[4];
  
  pyQCD::convertLinopParams(cuspIntParams, intParams,
			    cuspFloatParams, floatParams,
			    cuspComplexParams, complexParams);

  pyQCD::eigenToCusp(cuspBoundaryConditions, boundaryConditions);

  VectorTypeHost psiCusp(eta.size(), 0.0);
  VectorTypeHost etaCusp(eta.size(), 0.0);
  etaCusp = pyQCD::eigenToCusp(eta);

  pyQCD::invertDiracOperator(psiCusp, action, cuspIntParams, cuspFloatParams,
			     cuspComplexParams, cuspBoundaryConditions,
			     etaCusp, solverMethod, precondition, maxIterations,
			     tolerance, verbosity, this->spatialExtent,
			     this->temporalExtent,
			     gaugeField);

  psi = pyQCD::cuspToEigen(psiCusp);

  delete[] cuspIntParams;
  delete[] cuspFloatParams;
  delete[] cuspComplexParams;
  delete[] gaugeField;
			     
#else

  int iterations = maxIterations;
  double residual = tolerance;
  double time = 0.0;

  LinearOperator* diracMatrix;
  this->diracOperatorFactory(diracMatrix, action, intParams, floatParams,
			     complexParams, boundaryConditions);

  switch (solverMethod) {
  case pyQCD::bicgstab:
    psi = bicgstab(diracMatrix, eta, residual, iterations, time,
		   precondition);
    break;
  case pyQCD::cg:
    psi = cg(diracMatrix, eta, residual, iterations, time, precondition);
    break;
  case pyQCD::gmres:
    psi = gmres(diracMatrix, eta, residual, iterations, time, precondition);
    break;
  default:
    psi = cg(diracMatrix, eta, residual, iterations, time, precondition);
    break;	
  }
  if (verbosity > 0) {
    cout << "  -> Solver finished with residual of "
	 << residual << " in " << iterations << " iterations." << endl;
    cout << "  -> CPU time: " << time << " seconds" << endl;
  }

  delete diracMatrix;

#endif

  return psi;
}
