#include <pylattice.hpp>

pyLattice::pyLattice(const int nEdgePoints,
		     const double beta,
		     const double u0,
		     const int action,
		     const int nCorrelations,
		     const double rho,
		     const double epsilon, 
		     const int updateMethod,
		     const int parallelFlag) :
  Lattice::Lattice(nEdgePoints, beta, u0, action, nCorrelations, rho,
		   epsilon, updateMethod, parallelFlag)
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
				     const int nSmears)
{
  // Calculates the loop specified by corners c1 and c2 (which must
  // lie in the same plane)
  int tempCorner[4] = {py::extract<int>(corner[0]),
		       py::extract<int>(corner[1]),
		       py::extract<int>(corner[2]),
		       py::extract<int>(corner[3])};

  return this->computeWilsonLoop(tempCorner, r, t, dimension, nSmears);
}



double pyLattice::computeAverageWilsonLoopP(const int r, const int t,
					    const int nSmears)
{
  // Wrapper for the expectation value for the Wilson loop
  ScopedGILRelease scope;
  return this->computeAverageWilsonLoop(r, t, nSmears);
}



py::list pyLattice::computePropagatorP(const double mass, const py::list site,
				       const double spacing)
{
  // Wrapper for the calculation of a propagator
  int tempSite[4] = {py::extract<int>(site[0]),
		     py::extract<int>(site[1]),
		     py::extract<int>(site[2]),
		     py::extract<int>(site[3])};

  ScopedGILRelease scope;

  MatrixXcd prop = this->computePropagator(mass, tempSite, spacing);

  py::list pythonPropagator;
  
  for (int i = 0; i < 12; ++i) {
    py::list tempList;
    for (int j = 0; j < 12; ++j) {
      tempList.append(prop(i, j));
    }
    pythonPropagator.append(tempList);
  }

  return pythonPropagator;
}



py::list pyLattice::computeZeroMomPropagatorP(const double mass, const int time,
					      const double spacing)
{
  // Wrapper for the calculation of a propagator
  ScopedGILRelease scope;
  MatrixXcd prop = this->computeZeroMomPropagator(mass, time, spacing);

  py::list pythonPropagator;
  
  for (int i = 0; i < 12; ++i) {
    py::list tempList;
    for (int j = 0; j < 12; ++j) {
      tempList.append(prop(i, j));
    }
    pythonPropagator.append(tempList);
  }

  return pythonPropagator;
}



py::list pyLattice::computePropagatorsP(const double mass, const double spacing)
{
  // Wrapper for the calculation of propagators
  ScopedGILRelease scope;

  vector<MatrixXcd> props = this->computePropagators(mass, spacing);

  py::list pythonPropagators;
  int nSites = int(pow(this->nEdgePoints, 4));
  
  for (int i = 0; i < nSites; ++i) {
    py::list tempList1;
    for (int j = 0; j < 12; ++j) {
      py::list tempList2;
      for (int k = 0; k < 12; ++k) {
	tempList2.append(props[j, k]);
      }
      tempList1.append(tempList2);
    }
    pythonPropagators.append(tempList1);
  }

  return pythonPropagators;
}



void pyLattice::runThreads(const int nUpdates, const int remainder)
{
  ScopedGILRelease scope;
  Lattice::runThreads(nUpdates, remainder);
}



py::list pyLattice::getLinkP(const int n0, const int n1, const int n2,
			     const int n3, const int dimension) const
{
  // Returns the given link as a python nested list. Used in conjunction
  // with python interfaces library to extract the links as a nested list
  // of numpy matrices.
  py::list out;
  int index = dimension + 4
    * (n3 + this->nEdgePoints
       * (n2 + this->nEdgePoints
	  * (n1 + this->nEdgePoints * n0)));
  for (int i = 0; i < 3; i++) {
    py::list temp;
    for (int j = 0; j < 3; j++) {
      temp.append(this->links_[index](i, j));
    }
    out.append(temp);
  }
  return out;
}



py::list pyLattice::getRandSu3(const int index) const
{
  // Returns the given random SU3 matrix as a python list
  py::list out;
  for (int i = 0; i < 3; i++) {
    py::list temp;
    for (int j = 0; j < 3; j++) {
      temp.append(this->randSu3s_[index](i, j));
    }
    out.append(temp);
  }
  return out;
}
