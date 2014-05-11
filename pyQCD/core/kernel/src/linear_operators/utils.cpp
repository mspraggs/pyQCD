#include <linear_operators/utils.hpp>

namespace pyQCD
{
  VectorXcd multiplyGamma5(const VectorXcd& psi)
  {
    int nRows = psi.size();
    VectorXcd eta = VectorXcd::Zero(nRows);

#pragma omp parallel for
    for (int i = 0; i < nRows / 12; ++i)
      for (int j = 0; j < 4; ++j)
	for (int k = 0; k < 4; ++k)
	  for (int l = 0; l < 3; ++l)
	    eta(12 * i + 3 * j + l)
	      += gamma5(j, k) * psi(12 * i + 3 * k + l);

    return eta;    
  }



  VectorXcd multiplyPplus(const VectorXcd& psi)
  {
    int nRows = psi.size();
    VectorXcd eta = VectorXcd::Zero(nRows);

#pragma omp parallel for
    for (int i = 0; i < nRows / 12; ++i)
      for (int j = 0; j < 4; ++j)
	for (int k = 0; k < 4; ++k)
	  for (int l = 0; l < 3; ++l)
	    eta(12 * i + 3 * j + l)
	      += Pplus(j, k) * psi(12 * i + 3 * k + l);

    return eta;    
  }



  VectorXcd multiplyPminus(const VectorXcd& psi)
  {
    int nRows = psi.size();
    VectorXcd eta = VectorXcd::Zero(nRows);

#pragma omp parallel for
    for (int i = 0; i < nRows / 12; ++i)
      for (int j = 0; j < 4; ++j)
	for (int k = 0; k < 4; ++k)
	  for (int l = 0; l < 3; ++l)
	    eta(12 * i + 3 * j + l)
	      += Pminus(j, k) * psi(12 * i + 3 * k + l);

    return eta;    
  }



  vector<vector<int> > getNeighbourIndices(const int hopSize, const Lattice* lattice)
  {
    // Gets the site indices for the sites a certain number of hops away
    // from each of the sites on the lattice

    int numSites = lattice->spatialExtent * lattice->spatialExtent
      * lattice->spatialExtent * lattice->temporalExtent;

    int latticeShape[4] = {lattice->temporalExtent,
			   lattice->spatialExtent,
			   lattice->spatialExtent,
			   lattice->spatialExtent};

    vector<vector<int> > neighbourIndices;

    for (int i = 0; i < numSites; ++i) {

      vector<int> neighbours(8, 0);

      // Determine the coordinates of the site we're on
      int site[4]; // The coordinates of the lattice site
      getSiteCoords(i, lattice->spatialExtent,
		    lattice->temporalExtent, site);

      int siteBehind[4]; // Site/link index for the site/link behind us
      int siteAhead[4]; // Site/link index for the site/link in front of us

      // Loop over the four gamma indices (mu) in the sum inside the Wilson
      // action
      for (int mu = 0; mu < 4; ++mu) {
          
	// Now we determine the indices of the neighbouring links

	copy(site, site + 4, siteBehind);
	siteBehind[mu] = pyQCD::mod(siteBehind[mu] - hopSize, latticeShape[mu]);
	int siteBehindIndex = pyQCD::getSiteIndex(siteBehind,
						  latticeShape[1]);

	copy(site, site + 4, siteAhead);
	siteAhead[mu] = pyQCD::mod(siteAhead[mu] + hopSize, latticeShape[mu]);
	int siteAheadIndex = pyQCD::getSiteIndex(siteAhead,
						 latticeShape[1]);

	neighbours[mu] = siteBehindIndex;
	neighbours[mu + 4] = siteAheadIndex;
      }
      neighbourIndices.push_back(neighbours);
    }
    
    return neighbourIndices;
  }



  vector<vector<complex<double> > > getBoundaryConditions(
    const int hopSize, const vector<complex<double> >& boundaryConditions,
    const Lattice* lattice)
  {
    // Determines the boundary conditions for each hop

    int numSites = lattice->spatialExtent * lattice->spatialExtent
      * lattice->spatialExtent * lattice->temporalExtent;

    int latticeShape[4] = {lattice->temporalExtent,
			   lattice->spatialExtent,
			   lattice->spatialExtent,
			   lattice->spatialExtent};

    vector<vector<complex<double > > > output;

    for (int i = 0; i < numSites; ++i) {

      vector<complex<double> >
	siteBoundaryConditions(8, complex<double>(1.0, 0.0));

      // Determine the coordinates of the site we're on
      int site[4]; // The coordinates of the lattice site
      getSiteCoords(i, lattice->spatialExtent,
		    lattice->temporalExtent, site);

      // Loop over the four gamma indices (mu) in the sum inside the Wilson
      // action
      for (int mu = 0; mu < 4; ++mu) {
          
	// Now we determine the indices of the neighbouring links

	if (site[mu] - hopSize < 0 || site[mu] - hopSize >= latticeShape[mu])
	  siteBoundaryConditions[mu] = boundaryConditions[mu];

	if (site[mu] + hopSize < 0 || site[mu] + hopSize >= latticeShape[mu])
	  siteBoundaryConditions[mu + 4] = boundaryConditions[mu];
      }
      output.push_back(siteBoundaryConditions);
    }

    return output;
  }
}
