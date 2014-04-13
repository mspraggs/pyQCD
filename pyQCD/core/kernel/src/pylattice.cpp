#include <pylattice.hpp>

pyLattice::pyLattice(const int spatialExtent,
		     const int temporalExtent,
		     const double beta,
		     const double ut,
		     const double us,
		     const double chi,
		     const int action,
		     const int nCorrelations,
		     const int updateMethod,
		     const int parallelFlag,
		     const int chunkSize,
		     const int randSeed) :
  Lattice::Lattice(spatialExtent, temporalExtent, beta, ut, us, chi, action,
		   nCorrelations, updateMethod, parallelFlag, chunkSize,
		   randSeed)
{
  
}



pyLattice::pyLattice(const pyLattice& pylattice) : 
  Lattice::Lattice(pylattice)
{
  
}



pyLattice::~pyLattice()
{
  
}



double pyLattice::computePlaquetteP(const py::list site2, const int mu,
				    const int nu)
{
  // Python wrapper for the plaquette function.
  int site[4] = {py::extract<int>(site2[0]),
		 py::extract<int>(site2[1]),
		 py::extract<int>(site2[2]),
		 py::extract<int>(site2[3])};
  return this->computePlaquette(site, mu, nu);
}



double pyLattice::computeRectangleP(const py::list site, const int mu,
				    const int nu)
{
  // Python wrapper for rectangle function.
  int tempSite[4] = {py::extract<int>(site[0]),
		     py::extract<int>(site[1]),
		     py::extract<int>(site[2]),
		     py::extract<int>(site[3])};
  return this->computeRectangle(tempSite, mu, nu);
}



double pyLattice::computeTwistedRectangleP(const py::list site,
					   const int mu, const int nu)
{
  // Python wrapper for rectangle function
  int tempSite[4] = {py::extract<int>(site[0]),
		     py::extract<int>(site[1]),
		     py::extract<int>(site[2]),
		     py::extract<int>(site[3])};
  return this->computeTwistedRectangle(tempSite, mu, nu);
}



double pyLattice::computeWilsonLoopP(const py::list corner, const int r,
				     const int t, const int dimension,
				     const int nSmears,
				     const double smearingParameter)
{
  // Calculates the loop specified by corners c1 and c2 (which must
  // lie in the same plane)
  int tempCorner[4] = {py::extract<int>(corner[0]),
		       py::extract<int>(corner[1]),
		       py::extract<int>(corner[2]),
		       py::extract<int>(corner[3])};

  return this->computeWilsonLoop(tempCorner, r, t, dimension, nSmears,
				 smearingParameter);
}



double pyLattice::computeAverageWilsonLoopP(const int r, const int t,
					    const int nSmears,
					    const double smearingParameter)
{
  // Wrapper for the expectation value for the Wilson loop
  ScopedGILRelease scope;
  return this->computeAverageWilsonLoop(r, t, nSmears, smearingParameter);
}



py::list pyLattice::computeWilsonPropagatorP(
  const double mass, const py::list site,
  const int nSmears, const double smearingParameter,
  const int sourceSmearingType, const int nSourceSmears,
  const double sourceSmearingParameter, const int sinkSmearingType,
  const int nSinkSmears, const double sinkSmearingParameter,
  const int solverMethod, const py::list boundaryConditions,
  const int precondition, const int maxIterations,
  const double tolerance, const int verbosity)
{
  // Wrapper for the calculation of a propagator
  int tempSite[4];
  vector<complex<double> > tempBoundaryConditions;

  pyQCD::propagatorPrep(tempSite, tempBoundaryConditions, site,
			boundaryConditions);

  // Release the GIL for the propagator inversion
  ScopedGILRelease* scope = new ScopedGILRelease;
  // Get the propagator
  vector<MatrixXcd> prop 
    = this->computeWilsonPropagator(mass, tempSite, nSmears, smearingParameter,
				    sourceSmearingType, nSourceSmears,
				    sourceSmearingParameter, sinkSmearingType,
				    nSinkSmears, sinkSmearingParameter,
				    solverMethod, tempBoundaryConditions,
				    precondition, maxIterations, tolerance,
				    verbosity);
  // Put GIL back in place
  delete scope;

  return pyQCD::propagatorToList(prop);
}



py::list pyLattice::computeHamberWuPropagatorP(
  const double mass, const py::list site,
  const int nSmears, const double smearingParameter,
  const int sourceSmearingType, const int nSourceSmears,
  const double sourceSmearingParameter, const int sinkSmearingType,
  const int nSinkSmears, const double sinkSmearingParameter,
  const int solverMethod, const py::list boundaryConditions,
  const int precondition, const int maxIterations,
  const double tolerance, const int verbosity)
{
  // Wrapper for the calculation of a propagator
  int tempSite[4];
  vector<complex<double> > tempBoundaryConditions;

  pyQCD::propagatorPrep(tempSite, tempBoundaryConditions, site,
			boundaryConditions);
  // Release the GIL for the propagator inversion
  ScopedGILRelease* scope = new ScopedGILRelease;
  // Get the propagator
  vector<MatrixXcd> prop 
    = this->computeHamberWuPropagator(mass, tempSite, nSmears, smearingParameter,
				      sourceSmearingType, nSourceSmears,
				      sourceSmearingParameter, sinkSmearingType,
				      nSinkSmears, sinkSmearingParameter,
				      solverMethod, tempBoundaryConditions,
				      precondition, maxIterations, tolerance,
				      verbosity);
  // Put GIL back in place
  delete scope;

  return pyQCD::propagatorToList(prop);
}



py::list pyLattice::applyWilsonDiracOperator(py::list psi, const double mass,
					     py::list boundaryConditions,
					     const int precondition)
{
  // Apply the Wilson Dirac operator to the supplied vector/spinor

  VectorXcd vectorPsi = pyQCD::convertListToVector(psi);

  vector<complex<double> > tempBoundaryConditions
    = pyQCD::convertBoundaryConditions(boundaryConditions);

  LinearOperator* linop;

  // Release the GIL to apply the propagator
  ScopedGILRelease* scope = new ScopedGILRelease;

  // TODO: Case for precondition = 1
  linop = new Wilson(mass, tempBoundaryConditions, this);

  VectorXcd vectorEta = linop->apply(vectorPsi);

  delete linop;
  // Put the GIL back in place
  delete scope;

  return pyQCD::convertVectorToList(vectorEta);
}



py::list pyLattice::applyHamberWuDiracOperator(py::list psi, const double mass,
					       py::list boundaryConditions,
					       const int precondition)
{
  // Apply the Wilson Dirac operator to the supplied vector/spinor

  VectorXcd vectorPsi = pyQCD::convertListToVector(psi);

  vector<complex<double> > tempBoundaryConditions
    = pyQCD::convertBoundaryConditions(boundaryConditions);

  LinearOperator* linop;

  // Release the GIL to apply the propagator
  ScopedGILRelease* scope = new ScopedGILRelease;

  // TODO: Case for precondition = 1
  linop = new HamberWu(mass, tempBoundaryConditions, this);

  VectorXcd vectorEta = linop->apply(vectorPsi);

  delete linop;
  // Put the GIL back in place
  delete scope;

  return pyQCD::convertVectorToList(vectorEta);
}



py::list pyLattice::applyDWFDiracOperator(py::list psi, const double mass,
					  const double M5, const int Ls,
					  const int kernelType,
					  py::list boundaryConditions,
					  const int precondition)
{
  // Apply the Wilson Dirac operator to the supplied vector/spinor

  VectorXcd vectorPsi = pyQCD::convertListToVector(psi);

  vector<complex<double> > tempBoundaryConditions(4, complex<double>(1.0, 0.0));

  for (int i = 0; i < 4; ++i)
    tempBoundaryConditions[i] 
      = py::extract<complex<double> >(boundaryConditions[i]);

  // Release the GIL to apply the linear operator
  ScopedGILRelease* scope = new ScopedGILRelease;

  // TODO: Case for precondition = 1
  DWF linop(mass, M5, Ls, kernelType, tempBoundaryConditions, this);

  VectorXcd vectorEta = linop.apply(vectorPsi);

  // Put the GIL back in place
  delete scope;

  return pyQCD::convertVectorToList(vectorEta); 
}



py::list pyLattice::applyJacobiSmearingOperator(py::list psi,
						const int numSmears,
						const double smearingParameter,
						py::list boundaryConditions)
{
  // Apply the Wilson Dirac operator to the supplied vector/spinor

  VectorXcd vectorPsi = pyQCD::convertListToVector(psi);

  vector<complex<double> > tempBoundaryConditions(4, complex<double>(1.0, 0.0));

  for (int i = 0; i < 4; ++i)
    tempBoundaryConditions[i] 
      = py::extract<complex<double> >(boundaryConditions[i]);

  LinearOperator* linop;

  // Release the GIL to apply the propagator
  ScopedGILRelease* scope = new ScopedGILRelease;

  linop = new JacobiSmearing(numSmears, smearingParameter,
			     tempBoundaryConditions, this);

  VectorXcd vectorEta = linop->apply(vectorPsi);

  delete linop;
  // Put the GIL back in place
  delete scope;

  return pyQCD::convertVectorToList(vectorEta);
}



void pyLattice::runThreads(const int nUpdates, const int remainder)
{
  // Need to overload this and release the GIL
  ScopedGILRelease scope;
  Lattice::runThreads(nUpdates, remainder);
}



py::list pyLattice::getLinkP(const py::list link)
{
  // Returns the given link as a python nested list. Used in conjunction
  // with python interfaces library to extract the links as a nested list
  // of numpy matrices.
  int tempLink[5] = {py::extract<int>(link[0]),
		     py::extract<int>(link[1]),
		     py::extract<int>(link[2]),
		     py::extract<int>(link[3]),
		     py::extract<int>(link[4])};
  // Convert the Matrix3cd to a python list
  return pyQCD::convertMatrixToList(this->getLink(tempLink));
}



void pyLattice::setLinkP(const py::list link, const py::list matrix)
{
  // Set the given link to the values specified in matrix
  int tempLink[5] = {py::extract<int>(link[0]),
		     py::extract<int>(link[1]),
		     py::extract<int>(link[2]),
		     py::extract<int>(link[3]),
		     py::extract<int>(link[4])};
  Matrix3cd tempMatrix = pyQCD::convertListToMatrix(matrix);
  this->setLink(tempLink, tempMatrix);
}



py::list pyLattice::getRandSu3(const int index) const
{
  // Returns the given random SU3 matrix as a python list
  return pyQCD::convertMatrixToList(this->randSu3s_[index]);
}
