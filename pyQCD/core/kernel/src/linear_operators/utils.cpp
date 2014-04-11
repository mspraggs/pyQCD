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



  vector<vector<int> > getNeighbourIndices(const int hopSize, Lattice* lattice)
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
      int site[5]; // The coordinates of the lattice site
      getLinkCoords(4 * i, lattice->spatialExtent,
		     lattice->temporalExtent, site);

      int siteBehind[5]; // Site/link index for the site/link behind us
      int siteAhead[5]; // Site/link index for the site/link in front of us

      // Loop over the four gamma indices (mu) in the sum inside the Wilson
      // action
      for (int mu = 0; mu < 4; ++mu) {
          
	// Now we determine the indices of the neighbouring links

	copy(site, site + 5, siteBehind);
	siteBehind[mu] = pyQCD::mod(siteBehind[mu] - hopSize, latticeShape[mu]);
	int siteBehindIndex = pyQCD::getLinkIndex(siteBehind,
						  latticeShape[1]) / 4;

	copy(site, site + 5, siteAhead);
	siteAhead[mu] = pyQCD::mod(siteAhead[mu] + hopSize, latticeShape[mu]);
	int siteAheadIndex = pyQCD::getLinkIndex(siteAhead,
						 latticeShape[1]) / 4;

	neighbours[mu] = siteBehindIndex;
	neighbours[mu + 4] = siteAheadIndex;
      }
      neighbourIndices.push_back(neighbours);
    }
    
    return neighbourIndices;
  }



  vector<vector<complex<double> > > getBoundaryConditions(
							  const int hopSize, const vector<complex<double> >& boundaryConditions,
							  Lattice* lattice)
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
      int site[5]; // The coordinates of the lattice site
      getLinkCoords(4 * i, lattice->spatialExtent,
		     lattice->temporalExtent, site);

      // Loop over the four gamma indices (mu) in the sum inside the Wilson
      // action
      for (int mu = 0; mu < 4; ++mu) {
          
	// Now we determine the indices of the neighbouring links

	if (site[mu] - hopSize < 0 || site[mu] - hopSize >= latticeShape[mu])
	  siteBoundaryConditions[mu] = boundaryConditions[mu % 4];

	if (site[mu] + hopSize < 0 || site[mu] + hopSize >= latticeShape[mu])
	  siteBoundaryConditions[mu + 4] = boundaryConditions[mu % 4];
      }
      output.push_back(siteBoundaryConditions);
    }

    return output;
  }
}
