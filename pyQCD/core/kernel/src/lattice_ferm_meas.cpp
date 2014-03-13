#include <lattice.hpp>
#include <utils.hpp>

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
Lattice::computePropagator(const double mass, const double spacing, int site[4],
			   const int nSmears, const double smearingParameter,
			   const int nSourceSmears,
			   const double sourceSmearingParameter,
			   const int nSinkSmears,
			   const double sinkSmearingParameter,
			   const int solverMethod,
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

  // Get the dirac matrix
  if (verbosity > 0)
    cout << "  Generating Dirac matrix..." << flush;
  
  vector<complex<double> > boundaryConditions(4, complex<double>(1.0, 0.0));
  boundaryConditions[0] = complex<double>(-1.0, 0.0);

  // Create the source and sink smearing operators
  LinearOperator* sourceSmearingOperator 
    = new JacobiSmearing(nSourceSmears, sourceSmearingParameter,
			 boundaryConditions, this);
  LinearOperator* sinkSmearingOperator 
    = new JacobiSmearing(nSinkSmears, sinkSmearingParameter,
			 boundaryConditions, this);

  if (verbosity > 0)
    cout << " Done!" << endl;

  LinearOperator* diracMatrix 
    = new UnpreconditionedWilson(mass, boundaryConditions, this);

  // Loop through colour and spin indices and invert propagator
  for (int i = 0; i < 4; ++i) {
    for(int j = 0; j < 3; ++j) {
      if (verbosity > 0)
	cout << "  Inverting for spin " << i
	     << " and colour " << j << "..." << endl;
      // Create the source vector
      VectorXcd source = this->makeSource(site, i, j, sourceSmearingOperator);
      
      // Do the inversion
      double residual = 0.0;
      int iterations = 0;
      
      VectorXcd solution(3 * this->nLinks_);
      
      if (solverMethod == 1)
	solution = cg(diracMatrix, source, 1e-4, 1000, residual, iterations);
      else
	solution = bicgstab(diracMatrix, source, 1e-4, 1000, residual,
			    iterations);
      
      // Smear the sink
      solution = sinkSmearingOperator->apply(solution);
      
      // Add the result to the propagator matrix
      for (int k = 0; k < nSites; ++k) {
	for (int l = 0; l < 12; ++l) {
	  propagator[k](l, j + 3 * i) = solution(12 * k + l);
	}
      }
      if (verbosity > 0)
	cout << "  -> Inversion reached tolerance of "
	     << residual << " in " << iterations
	     << " iterations." << endl;
    }
  }

  delete diracMatrix;
  delete sourceSmearingOperator;
  delete sinkSmearingOperator;

  // Restore the non-smeared gauge field
  if (nSmears > 0)
    this->links_ = templinks;

  return propagator;
}
