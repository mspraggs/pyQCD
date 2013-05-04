#include "lattice.cpp"
#include "pylattice.hpp"

pyLattice::pyLattice(const int nEdgePoints,
		     const double beta,
		     const int nCorrelations,
		     const int nConfigurations,
		     const double epsilon,
		     const double a,
		     const double rho,
		     const double u0,
		     const int action) :
  Lattice::Lattice(nEdgePoints, beta, nCorrelations, nConfigurations,
		   epsilon, a, rho, u0, action)
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
  //Python wrapper for rectangle function.
  int site[4] = {py::extract<int>(site2[0]),
		 py::extract<int>(site2[1]),
		 py::extract<int>(site2[2]),
		 py::extract<int>(site2[3])};
  return this->computeRectangle(site, mu, nu);
}



double pyLattice::computeTwistedRectangleP(const py::list site,
					   const int mu, const int nu)
{
  // Python wrapper for rectangle function
  int tempSite[4] = {py::extract<int>(site[0]),
		     py::extract<int>(site[1]),
		     py::extract<int>(site[2]),
		     py::extract<int>(site[3])};
  return this->computeTwistRect(tempSite, mu, nu);
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
  return Lattice::computeAverageWilson(r, t, nSmears);
}



void pyLattice::runThreads(const int size, const int nUpdates,
			   const int remainder)
{
  ScopedGILRelease scope = ScopedGILRelease();
  Lattice::runThreads(size, nUpdates, remainder);
}



py::list pyLattice::getLinkP(const int n1, const int n2, const int n3,
			     const int n4, const int dimension) const
{
  // Returns the given link as a python nested list. Used in conjunction
  // with python interfaces library to extract the links as a nested list
  // of numpy matrices.
  py::list out;
  for (int i = 0; i < 3; i++) {
    py::list temp;
    for (int j = 0; j < 3; j++) {
      temp.append(this->links[n0][n1][n2][n3][dimension](i, j));
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
      temp.append(this->randSU3s[index](i, j));
    }
    out.append(temp);
  }
  return out;
}
