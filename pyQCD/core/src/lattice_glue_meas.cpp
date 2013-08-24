#include <lattice.hpp>
#include <utils.hpp>


Matrix3cd Lattice::computePath(const vector<vector<int> >& path)
{
  // Multiplies the matrices together specified by the indices in path
  Matrix3cd out = Matrix3cd::Identity();
  
  for (int i = 0; i < path.size() - 1; ++i) {
    // Which dimension are we moving in?
    int dimension = path[i][4];
    int dimensionDifference = path[i + 1][dimension] - path[i][dimension];

    if (abs(dimensionDifference) != 1) {
      // Consecutive points don't match link direction, so throw an error
      cout << "Error! Path contains non-consecutive link variables." << endl;
    }
    else if (dimensionDifference == -1) {
      // We're going backwards, so the link must be the adjoint of the link 
      // matrix, which we get by using the next site on the lattice.
      int link[5] = {path[i + 1][0],
		     path[i + 1][1],
		     path[i + 1][2],
		     path[i + 1][3],
		     path[i][4]};

      out *= this->getLink(link).adjoint();
    }
    else {
      // We're going forwards, so it's just the normal matrix
      int link[5] = {path[i][0],
		     path[i][1],
		     path[i][2],
		     path[i][3],
		     path[i][4]};

      out *= this->getLink(link);
    }
  }
  return out;
}



Matrix3cd Lattice::computeLine(const int start[4], const int finish[4])
{
  // Multiplies all gauge links along line from start to finish
  // First check that's actually a straight path
  int countDimensions = 0;
  int dimension = 0;
  for (int i = 0; i < 4; ++i) {
    if (abs(start[i] - finish[i]) != 0) {
      // Keep track of the most recent dimension that differs between start
      // and finish, as if we're good to go then we'll need this when
      // defining the path.
      dimension = i;
      countDimensions++;
    }
  }

  if (countDimensions != 1) {
    cout << "Error! Start and end points do not form a straight line."
	 << endl;
    return Matrix3cd::Identity();
  }
  else {
    // If the two points are on the same line, we're good to go
    vector<vector<int> >  line; //Stores the path

    // Now need to know if we're going backwards or forwards
    if (start[dimension] > finish[dimension]) {
      for (int i = start[dimension]; i >= finish[dimension]; i--) {
	// Create the link vector to append, and initialize it's elements
	vector<int> link;
	link.assign(start, start + 4);
	// Update the index that's parallel to the line with the current
	// location
	link[dimension] = i;
	// The direction is going to be equal to the direction of the line
	link.push_back(dimension);
	// Push it onto the line
	line.push_back(link);
      }
      return this->computePath(line);
    }
    else {
      // Same again, but this time we deal with the case of going backwards
      for (int i = start[dimension]; i <= finish[dimension]; ++i) {
	vector<int> link;
	link.assign(start, start + 4);
	link[dimension] = i;
	link.push_back(dimension);
	line.push_back(link);
      }
      return this->computePath(line);
    }
  }
}



double Lattice::computeWilsonLoop(const int corner1[4], const int corner2[4],
				  const int nSmears,
				  const double smearingParameter)
{
  // Calculates the loop specified by corners corner1 and corner2 (which must
  // lie in the same plane)
  Matrix3cd out = Matrix3cd::Identity();
  GaugeField linkStore1;
  GaugeField linkStore2;
  // Smear the links if specified, whilst storing the non-smeared ones.
  int timeSliceIncrement = this->nLinks_ / this->temporalExtent;
  if (nSmears > 0) {
    linkStore1.resize(timeSliceIncrement);
    linkStore2.resize(timeSliceIncrement);
    int corner1Timeslice = pyQCD::mod(corner1[0], this->temporalExtent);
    int corner2Timeslice = pyQCD::mod(corner2[0], this->temporalExtent);
    
    for (int i = 0; i < timeSliceIncrement; ++i) {
      linkStore1[i] = this->links_[corner1Timeslice + i];
      linkStore2[i] = this->links_[corner2Timeslice + i];
    }
    this->smearLinks(corner1[0], nSmears, smearingParameter);
    this->smearLinks(corner2[0], nSmears, smearingParameter);
  }

  // Check that corner1 and corner2 are on the same plane
  int dimensionCount = 0;
  for (int i = 0; i < 4; ++i) {
    if (corner1[i] != corner2[i]) {
      dimensionCount++;
    }
  }
  
  if (dimensionCount != 2 || corner1[0] == corner2[0]) {
    cout << "Error! The two corner points do not form a rectangle with"
	 << " two spatial and two temporal sides." << endl;
  }
  else {
    // Get the second corner (going round the loop)
    int corner3[4] = {corner1[0], corner1[1], corner1[2], corner1[3]};
    corner3[0] = corner2[0];
    // Calculate the line segments between the first three corners
    out *= this->computeLine(corner1, corner3);
    out *= this->computeLine(corner3, corner2);
    // And repeat for the second set of sides
    int corner4[4] = {corner2[0], corner2[1], corner2[2], corner2[3]};
    corner4[0] = corner1[0];
    out *= this->computeLine(corner2, corner4);
    out *= this->computeLine(corner4, corner1);
  }
  // Restore the old links
  if (nSmears > 0) {
    int corner1Timeslice = pyQCD::mod(corner1[0], this->temporalExtent);
    int corner2Timeslice = pyQCD::mod(corner2[0], this->temporalExtent);
    for (int i = 0; i < timeSliceIncrement; ++i) {
      this->links_[corner1Timeslice + i] = linkStore1[i];
      this->links_[corner2Timeslice + i] = linkStore2[i];
    }
  }
  
  return out.trace().real() / 3.0;
}



double Lattice::computeWilsonLoop(const int corner[4], const int r,
				  const int t, const int dimension,
				  const int nSmears,
				  const double smearingParameter)
{
  // Calculates the loop specified by initial corner, width, height and
  // dimension

  GaugeField linkStore1;
  GaugeField linkStore2;
  // Smear the links if specified, whilst storing the non-smeared ones.
  int timeSliceIncrement = this->nLinks_ / this->temporalExtent;
  if (nSmears > 0) {
    linkStore1.resize(timeSliceIncrement);
    linkStore2.resize(timeSliceIncrement);
    int corner1Timeslice = pyQCD::mod(corner[0], this->temporalExtent);
    int corner2Timeslice = pyQCD::mod(corner[0] + t, this->temporalExtent);
    
    for (int i = 0; i < timeSliceIncrement; ++i) {
      linkStore1[i] = this->links_[corner1Timeslice + i];
      linkStore2[i] = this->links_[corner2Timeslice + i];
    }
    this->smearLinks(corner[0], nSmears, smearingParameter);
    this->smearLinks(corner[0] + t, nSmears, smearingParameter);
  }
  // An output matrix
  Matrix3cd out = Matrix3cd::Identity();
  int link[5];
  copy(corner, corner + 4, link);
  link[4] = dimension;

  // Calculate the first spatial edge
  for (int i = 0; i < r; ++i) {
    out *= this->getLink(link);
    link[dimension]++;
  }
  
  link[4] = 0;
  
  // Calculate the first temporal edge
  for (int i = 0; i < t; ++i) {
    out *= this->getLink(link);
    link[0]++;
  }
  
  link[4] = dimension;

  // Calculate the second spatial edge
  for (int i = 0; i < r; ++i) {
    link[dimension]--;
    out *= this->getLink(link).adjoint();
  }

  link[4] = 0;

  // Calculate the second temporal edge
  for (int i = 0; i < t; ++i) {
    link[0]--;
    out *= this->getLink(link).adjoint();
  }

  // Restore the old links
  if (nSmears > 0) {
    int corner1Timeslice = pyQCD::mod(corner[0], this->temporalExtent);
    int corner2Timeslice = pyQCD::mod(corner[0] + t, this->temporalExtent);
    for (int i = 0; i < timeSliceIncrement; ++i) {
      this->links_[corner1Timeslice + i] = linkStore1[i];
      this->links_[corner2Timeslice + i] = linkStore2[i];
    }
  }

  return out.trace().real() / 3.0;
}


double Lattice::computePlaquette(const int site[4], const int mu,
				 const int nu)
{
  // Calculate the plaquette operator at the given site, on plaquette
  // specified by directions mu and nu.

  // We define some variables to contain the offsets of the various
  // links around the lattice
  int mu_vec[4] = {0, 0, 0, 0};
  mu_vec[mu] = 1;
  int nu_vec[4] = {0, 0, 0, 0};
  nu_vec[nu] = 1;
  // The links also contain direction information, so we must create a new
  // set of variables to keep track of the directions of the links.
  int link1[5] = {0, 0, 0, 0, mu};
  int link2[5] = {0, 0, 0, 0, nu};
  int link3[5] = {0, 0, 0, 0, mu};
  int link4[5] = {0, 0, 0, 0, nu};

  // Do some assignment
  for (int i = 0; i < 4; ++i) {
    link1[i] = site[i];
    link2[i] = site[i] + mu_vec[i];
    link3[i] = site[i] + nu_vec[i];
    link4[i] = site[i];
  }

  // Run through the links and multiply them together.
  Matrix3cd product = this->getLink(link1);
  product *= this->getLink(link2);
  product *= this->getLink(link3).adjoint();
  product *= this->getLink(link4).adjoint();
  return product.trace().real() / 3.0;
}



double Lattice::computeRectangle(const int site[4], const int mu,
				 const int nu)
{
  // Calculate the rectangle operator at the given site, on the rectangle
  // specified by directions mu and nu.
  
  //Define some variables to offset the given site
  int mu_vec[4] = {0, 0, 0, 0};
  mu_vec[mu] = 1;
  int nu_vec[4] = {0, 0, 0, 0};
  nu_vec[nu] = 1;
  // Links also contain direction information, so must create a new set of
  // variables
  int link1[5] = {0, 0, 0, 0, mu};
  int link2[5] = {0, 0, 0, 0, mu};
  int link3[5] = {0, 0, 0, 0, nu};
  int link4[5] = {0, 0, 0, 0, mu};
  int link5[5] = {0, 0, 0, 0, mu};
  int link6[5] = {0, 0, 0, 0, nu};

  for (int i = 0; i < 4; ++i) {
    link1[i] = site[i];
    link2[i] = site[i] + mu_vec[i];
    link3[i] = site[i] + 2 * mu_vec[i];
    link4[i] = site[i] + mu_vec[i] + nu_vec[i];
    link5[i] = site[i] + nu_vec[i];
    link6[i] = site[i];
  }

  // Multiply all the links together to get the product.
  Matrix3cd product = this->getLink(link1);
  product *= this->getLink(link2);
  product *= this->getLink(link3);
  product *= this->getLink(link4).adjoint();
  product *= this->getLink(link5).adjoint();
  product *= this->getLink(link6).adjoint();
  
  return product.trace().real() / 3.0;  
}



double Lattice::computeTwistedRectangle(const int site[4], const int mu,
					const int nu)
{
  // Calculate the twisted rectangle operator
  
  // Define some variables to offset the given site
  int mu_vec[4] = {0, 0, 0, 0};
  mu_vec[mu] = 1;
  int nu_vec[4] = {0, 0, 0, 0};
  nu_vec[nu] = 1;
  // Links also contain direction information, so must create a new set of
  // variables
  int link1[5] = {0, 0, 0, 0, mu};
  int link2[5] = {0, 0, 0, 0, nu};
  int link3[5] = {0, 0, 0, 0, mu};
  int link4[5] = {0, 0, 0, 0, nu};
  int link5[5] = {0, 0, 0, 0, mu};
  int link6[5] = {0, 0, 0, 0, nu};
  int link7[5] = {0, 0, 0, 0, mu};
  int link8[5] = {0, 0, 0, 0, nu};

  for (int i = 0; i < 4; ++i) {
    link1[i] = site[i];
    link2[i] = site[i] + mu_vec[i];
    link3[i] = site[i] + mu_vec[i] + nu_vec[i];
    link4[i] = site[i] + 2 * mu_vec[i];
    link5[i] = site[i] + mu_vec[i];
    link6[i] = site[i] + mu_vec[i];
    link7[i] = site[i] + nu_vec[i];
    link8[i] = site[i];
  }

  // Multiply all the links together to get the product.
  Matrix3cd product = this->getLink(link1);
  product *= this->getLink(link2);
  product *= this->getLink(link3);
  product *= this->getLink(link4).adjoint();
  product *= this->getLink(link5).adjoint();
  product *= this->getLink(link6);
  product *= this->getLink(link7).adjoint();
  product *= this->getLink(link8).adjoint();

  return product.trace().real() / 3.0;
}



double Lattice::computeAveragePlaquette()
{
  // Calculate average plaquette operator value
  // mu > nu, so there are six plaquettes at each site.
  int nus[6] = {0, 0, 0, 1, 1, 2};
  int mus[6] = {1, 2, 3, 2, 3, 3};
  double Ptot = 0.0;
  // Pretty simple: step through the matrix and add all plaquettes up
  for (int i = 0; i < this->temporalExtent; ++i) {
    for (int j = 0; j < this->spatialExtent; ++j) {
      for (int k = 0; k < this->spatialExtent; ++k) {
	for (int l = 0; l < this->spatialExtent; ++l) {
	  for (int m = 0; m < 6; ++m) {
	    int site[4] = {i, j, k, l};
	    Ptot += this->computePlaquette(site,mus[m],nus[m]);
	  }
	}
      }
    }
  }
  // Divide through by number of plaquettes to get mean (simples!)
  return Ptot / (pow(this->spatialExtent, 3) * this->temporalExtent * 6);
}



double Lattice::computeAverageRectangle()
{
  // Calculate average plaquette operator value
  // mu > nu, so there are six plaquettes at each site.
  int nus[6] = {0, 0, 0, 1, 1, 2};
  int mus[6] = {1, 2, 3, 2, 3, 3};
  double Rtot = 0.0;
  // Pretty simple: step through the matrix and add all plaquettes up
  for (int i = 0; i < this->temporalExtent; ++i) {
    for (int j = 0; j < this->spatialExtent; ++j) {
      for (int k = 0; k < this->spatialExtent; ++k) {
	for (int l = 0; l < this->spatialExtent; ++l) {
	  for (int m = 0; m < 6; ++m) {
	    int site[4] = {i, j, k, l};
	    Rtot += this->computeRectangle(site, mus[m], nus[m]);
	  }
	}
      }
    }
  }
  // Divide through by number of plaquettes to get mean (simples!)
  return Rtot / (pow(this->spatialExtent, 3) * this->temporalExtent * 6);
}



double Lattice::computeAverageWilsonLoop(const int r, const int t,
					 const int nSmears,
					 const double smearingParameter)
{
  // Calculates the average of all possible Wilson loops of a given
  // dimension.
  // First off, save the current links and smear all time slices
  GaugeField templinks;
  if (nSmears > 0) {
    templinks = this->links_;
    for (int time = 0; time < this->temporalExtent; time++) {
      this->smearLinks(time, nSmears, smearingParameter);
    }
  }

  double Wtot = 0.0;
  if (this->parallelFlag_ == 1) {
#pragma omp parallel for collapse(5) reduction(+ : Wtot)
    for (int i = 0; i < this->temporalExtent; ++i) {
      for (int j = 0; j < this->spatialExtent; ++j) {
	for (int k = 0; k < this->spatialExtent; ++k) {
	  for (int l = 0; l < this->spatialExtent; ++l) {
	    for (int m = 1; m < 4; ++m) {
	      int site[4] = {i, j, k, l};
	      // Note, running in parallel causes very
	      // small variations in the final value
	      // of Wtot between consecutive calls
	      // (of the order of 10^-16)
	      Wtot += this->computeWilsonLoop(site, r, t, m, 0, 1.0);
	    }
	  }
	}
      }
    }
  }
  else {
    for (int i = 0; i < this->temporalExtent; ++i) {
      for (int j = 0; j < this->spatialExtent; ++j) {
	for (int k = 0; k < this->spatialExtent; ++k) {
	  for (int l = 0; l < this->spatialExtent; ++l) {
	    for (int m = 1; m < 4; ++m) {
	      int site[4] = {i, j, k, l};
	      Wtot += this->computeWilsonLoop(site, r, t, m, 0, 1.0);
	    }
	  }
	}
      }
    }
  }
  if (nSmears > 0)
    this->links_ = templinks;
  return Wtot / (pow(this->spatialExtent, 3) * this->temporalExtent * 3);
}



double Lattice::computeMeanLink()
{
  // Pretty simple: step through the matrix and add all link traces up
  double totalLink = 0;
  for (int i = 0; i < this->nLinks_; ++i) {
    totalLink += 1.0 / 3.0
      * this->links_[i].trace().real();
  }
  return totalLink / this->nLinks_;
}



Matrix3cd Lattice::computeQ(const int link[5], const double smearingParameter)
{
  // Calculates Q matrix for analytic smearing according to paper on analytic
  // smearing (Morningstart and Peardon, 2003)
  Matrix3cd C = Matrix3cd::Zero();

  for (int nu = 1; nu < 4; nu++) {
    if (nu != link[4]) {
      int tempLink[5] = {0, 0, 0, 0, 0};
      copy(link, link + 4, tempLink);
      tempLink[4] = nu;
      Matrix3cd tempMatrix = this->getLink(tempLink);

      tempLink[4] = link[4];
      tempLink[nu] += 1;
      tempMatrix *= this->getLink(tempLink);

      tempLink[4] = nu;
      tempLink[nu] -= 1;
      tempLink[link[4]] += 1;
      tempMatrix *= this->getLink(tempLink).adjoint();

      C += tempMatrix;

      copy(link, link + 4, tempLink);
      tempLink[nu] -= 1;
      tempLink[4] = nu;
      tempMatrix = this->getLink(tempLink).adjoint();

      tempLink[4] = link[4];
      tempMatrix *= this->getLink(tempLink);

      tempLink[4] = nu;
      tempLink[link[4]] += 1;
      tempMatrix *= this->getLink(tempLink);

      C += tempMatrix;
    }
  }
  
  C *= smearingParameter;

  Matrix3cd Omega = C * this->getLink(link).adjoint();
  Matrix3cd OmegaAdjoint = Omega.adjoint() - Omega;
  Matrix3cd out = 0.5 * pyQCD::i * OmegaAdjoint;
  return out - pyQCD::i / 6.0 * OmegaAdjoint.trace() * Matrix3cd::Identity();
}



void Lattice::smearLinks(const int time, const int nSmears,
			 const double smearingParameter)
{
  // Smear the specified time slice by iterating calling this function
  int nSpatialLinks = this->nLinks_ / this->temporalExtent;

  if (this->parallelFlag_ == 1) {

    for (int i = 0; i < nSmears; ++i) {
      // Iterate through all the links and calculate the new ones from
      // the existing ones.    
      GaugeField newLinks(nSpatialLinks);
#pragma omp parallel for
      for (int j = 0; j < nSpatialLinks; j+=4) {
	// NB, spatial links only, so l > 0!
	newLinks[j] = this->links_[time * nSpatialLinks + j];
	for (int k = 1; k < 4; ++k) {
	  // Create a temporary matrix to store the new link
	  int link[5];
	  pyQCD::getLinkIndices(time * nSpatialLinks + j + k,
				this->spatialExtent, this->temporalExtent, link);
	  Matrix3cd tempMatrix = this->computeQ(link, smearingParameter);
	  newLinks[j + k] = (pyQCD::i * tempMatrix).exp() * this->getLink(link);
	}
      }
      // Apply the changes to the existing lattice.
      for (int i = 0; i < nSpatialLinks; ++i) {
	this->links_[time * nSpatialLinks + i] = newLinks[i];
      }
    }
  }
  else {

    for (int i = 0; i < nSmears; ++i) {
      // Iterate through all the links and calculate the new ones from
      // the existing ones.    
      GaugeField newLinks(nSpatialLinks);

      for (int j = 0; j < nSpatialLinks; j+=4) {
	// NB, spatial links only, so l > 0!
	newLinks[j] = this->links_[time * nSpatialLinks + j];
	for (int k = 1; k < 4; ++k) {
	  // Create a temporary matrix to store the new link
	  int link[5];
	  pyQCD::getLinkIndices(time * nSpatialLinks + j + k,
				this->spatialExtent, this->temporalExtent, link);
	  Matrix3cd tempMatrix = this->computeQ(link, smearingParameter);
	  newLinks[time * nSpatialLinks + j + k] = (pyQCD::i * tempMatrix).exp()
	    * this->getLink(link);
	}
      }
      // Apply the changes to the existing lattice.
      for (int i = 0; i < nSpatialLinks; ++i) {
	this->links_[time * nSpatialLinks + i] = newLinks[i];
      }
    }
  }
}
