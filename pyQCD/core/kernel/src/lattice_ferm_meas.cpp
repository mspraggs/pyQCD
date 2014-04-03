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



vector<MatrixXcd>
Lattice::computeWilsonPropagator(
  const double mass, int site[4], const int nSmears,
  const double smearingParameter, const int sourceSmearingType,
  const int nSourceSmears, const double sourceSmearingParameter,
  const int sinkSmearingType, const int nSinkSmears,
  const double sinkSmearingParameter, const int solverMethod,
  const vector<complex<double> >& boundaryConditions, const int precondition,
  const int maxIterations, const double tolerance, const int verbosity)
{
  // Computes the Wilson propagator given a specified mass, source site
  // and set of smearing parameters

  // Generate the dirac matrix
  if (verbosity > 0)
    cout << "  Generating Dirac matrix..." << flush;

  LinearOperator* diracOperator;

  // TODO: Preconditioned Wilson operator (odd/even)
  // If we require preconditioning, create the preconditioned operator
  diracOperator = new UnpreconditionedWilson(mass, boundaryConditions, this);

  if (verbosity > 0)
    cout << " Done!" << endl;

  vector<MatrixXcd> propagator 
    = this->computePropagator(diracOperator, site, nSmears, smearingParameter,
			      sourceSmearingType, nSourceSmears,
			      sourceSmearingParameter, sinkSmearingType,
			      nSinkSmears, sinkSmearingParameter, solverMethod,
			      maxIterations, tolerance, verbosity);

  delete diracOperator;

  return propagator;
}



vector<MatrixXcd>
Lattice::computePropagator(LinearOperator* diracMatrix, int site[4],
			   const int nSmears, const double smearingParameter,
			   const int sourceSmearingType,
			   const int nSourceSmears,
			   const double sourceSmearingParameter,
			   const int sinkSmearingType,
			   const int nSinkSmears,
			   const double sinkSmearingParameter,
			   const int solverMethod,
			   const int maxIterations,
			   const double tolerance,
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
  
  vector<complex<double> > boundaryConditions(4, complex<double>(1.0, 0.0));
  boundaryConditions[0] = complex<double>(-1.0, 0.0);

  // Create the source and sink smearing operators
  // TODO: If statements for different types of smearing/sources
  LinearOperator* sourceSmearingOperator 
    = new JacobiSmearing(nSourceSmears, sourceSmearingParameter,
			 boundaryConditions, this);
  LinearOperator* sinkSmearingOperator 
    = new JacobiSmearing(nSinkSmears, sinkSmearingParameter,
			 boundaryConditions, this);

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
      case 0:
	solution = bicgstab(diracMatrix, source, residual, iterations, time);
	break;
      case 1:
	solution = cg(diracMatrix, source, residual, iterations, time);
	break;
      case 2:
	solution = gmres(diracMatrix, source, residual, iterations, time);
	break;
      default:
	solution = cg(diracMatrix, source, residual, iterations, time);
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

  // Restore the non-smeared gauge field
  if (nSmears > 0)
    this->links_ = templinks;

  return propagator;
}
