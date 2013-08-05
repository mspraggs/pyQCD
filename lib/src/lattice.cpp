#include <lattice.hpp>
#include <pyQCD_utils.hpp>

Lattice::Lattice(const int nEdgePoints, const double beta, const double u0,
		 const int action, const int nCorrelations, const double rho,
		 const int updateMethod, const int parallelFlag)
{
  // Default constructor. Assigns function arguments to member variables
  // and initializes links.
  this->nEdgePoints = nEdgePoints;
  this->nLinks_ = int(pow(this->nEdgePoints, 4) * 4);
  this->beta_ = beta;
  this->nCorrelations = nCorrelations;
  this->rho_ = rho;
  this->nUpdates_ = 0;
  this->u0_ = u0;
  this->action_ = action;
  this->updateMethod_ = updateMethod;
  this->parallelFlag_ = parallelFlag;

  // Initialize parallel Eigen
  initParallel();

  // Resize the link vector and assign each link a random SU3 matrix
  // Also set up the propagatorColumns vector
  this->links_.resize(this->nLinks_);
  this->propagatorColumns_ = vector<vector<vector<int> > >(this->nLinks_ / 4,
    vector<vector<int> >(8,vector<int>(2,0)));

  for (int i = 0; i < this->nLinks_; ++i)
    this->links_[i] = Matrix3cd::Identity();//this->makeRandomSu3();

  // Loop through the columns of the propagator and add any non-zero entries to
  // the propagatorColumns vector (maybe move the following to the utils file
  // as a function, in case next-to nearest neighbours are ever needed).

  // First define some offsets
  int offsets[8][4] = {{-1,0,0,0},{0,-1,0,0},{0,0,-1,0},{0,0,0,-1},
		       {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};

  // Loop through all site indices and add all nearest-neighbour indices to
  // the list

  for (int i = 0; i < this->nLinks_; i += 4) {
    
    // Initialize the relevant site vector for the row index
    int rowLink[5];
    pyQCD::getLinkIndices(i, this->nEdgePoints, rowLink);
   
    // Something to hold the neighbours for the row temporarily
    vector<int> rowNeighbours;
    // Something to keep track of the index in the sub vector
    int rowNeighboursIndex = 0;

    // Inner loop for the sites

    for (int j = 0; j < this->nLinks_; j += 4) {
      // Get the coordinates for the column
      int columnLink[5];
      pyQCD::getLinkIndices(j, this->nEdgePoints, columnLink);

      int neighbourCount = 0;
      int dimension = 0;

      // Look and see if any offsets apply
      for (int k = 0; k < 8; ++k) {
	bool isNeighbour = true;

	// Loop through the coordiates for the pair of sites and see if they're
	// neighbours
	for (int l = 0; l < 4; ++l) {
	  if (rowLink[l] != pyQCD::mod(columnLink[l] + offsets[k][l],
				       this->nEdgePoints)) {
	    isNeighbour = false;
	    break;
	  }
	}
	// If the sites are neighbours, the dimension that they lie in will be
	// needed below.
	if (isNeighbour) {
	  neighbourCount++;
	  dimension = k;
	}
      }
      // If the neighbour exists, add it to the list.
      if (neighbourCount > 0) {
	this->propagatorColumns_[i / 4][rowNeighboursIndex][0] = j;
	this->propagatorColumns_[i / 4][rowNeighboursIndex][1] = dimension;
	rowNeighboursIndex++;
      }
    }
  }

  // Generate a set of random SU3 matrices for use in the updates
  for (int i = 0; i < 200; ++i) {
    Matrix3cd randSu3 = this->makeRandomSu3();
    this->randSu3s_.push_back(randSu3);
    this->randSu3s_.push_back(randSu3.adjoint());
  }

  // Set the action to point to the correct function
  if (action == 0) {
    this->computeLocalAction = &Lattice::computeLocalWilsonAction;
    this->computeStaples = &Lattice::computeWilsonStaples;
  }
  else if (action == 1) {
    this->computeLocalAction = &Lattice::computeLocalRectangleAction;
    this->computeStaples = &Lattice::computeRectangleStaples;
  }
  else if (action == 2) {
    this->computeLocalAction = &Lattice::computeLocalTwistedRectangleAction;
    this->computeStaples = &Lattice::computeTwistedRectangleStaples;
    cout << "Warning! Heatbath updates are not implemented for twisted"
	 << " rectangle operator" << endl;
  }
  else {
    cout << "Warning! Specified action does not exist." << endl;
    this->computeLocalAction = &Lattice::computeLocalWilsonAction;
    this->computeStaples = &Lattice::computeWilsonStaples;
  }

  // Set the update method to point to the correct function
  if (updateMethod == 0) {
    if (action != 2) {
      this->updateFunction_ = &Lattice::heatbath;
    }
    else {
      cout << "Warning! Heatbath updates are not compatible with twisted "
	   << "rectangle action. Using Monte Carlo instead" << endl;
      this->updateFunction_ = &Lattice::monteCarloNoStaples;
    }
  }
  else if (updateMethod == 1) {
    if (action != 2) {
      this->updateFunction_ = &Lattice::monteCarlo;
    }
    else {
      cout << "Warning! Heatbath updates are not compatible with twisted "
	   << "rectangle action. Using Monte Carlo instead" << endl;
      this->updateFunction_ = &Lattice::monteCarloNoStaples;
    }
  }
  else if (updateMethod == 2) {
    this->updateFunction_ = &Lattice::monteCarloNoStaples;
  }
  else {
    cout << "Warning! Specified update method does not exist!" << endl;
    if (action != 2) {
      this->updateFunction_ = &Lattice::heatbath;
    }
    else {
      cout << "Warning! Heatbath updates are not compatible with twisted "
	   << "rectangle action. Using Monte Carlo instead" << endl;
      this->updateFunction_ = &Lattice::monteCarloNoStaples;
    }
  }

  // Initialize series of offsets used when doing block updates

  int chunkSize = 4;
  for (int i = 0; i < chunkSize; ++i) {
    for (int j = 0; j < chunkSize; ++j) {
      for (int k = 0; k < chunkSize; ++k) {
	for (int l = 0; l < chunkSize; ++l) {
	  for (int m = 0; m < 4; ++m) {
	    // We'll need an array with the link indices
	    int index = pyQCD::getLinkIndex(i, j, k, l, m, this->nEdgePoints);
	    this->chunkSequence_.push_back(index);
	  }
	}
      }
    }
  }

  int nChunks = int(pow(this->nEdgePoints / chunkSize, 4));

  for (int i = 0; i < this->nEdgePoints; i += chunkSize) {
    for (int j = 0; j < this->nEdgePoints; j += chunkSize) {
      for (int k = 0; k < this->nEdgePoints; k += chunkSize) {
	for (int l = 0; l < this->nEdgePoints; l += chunkSize) {
	  // We'll need an array with the link indices
	  int index = pyQCD::getLinkIndex(i, j, k, l, 0, this->nEdgePoints);
	  if (((i + j + k + l) / chunkSize) % 2 == 0)
            this->evenBlocks_.push_back(index);
	  else
	    this->oddBlocks_.push_back(index);
	}
      }
    }
  }
}



Lattice::Lattice(const Lattice& lattice)
{
  // Default constructor. Assigns function arguments to member variables
  // and initializes links.
  this->nEdgePoints = lattice.nEdgePoints;
  this->nLinks_ = lattice.nLinks_;
  this->beta_ = lattice.beta_;
  this->nCorrelations = lattice.nCorrelations;
  this->rho_ = lattice.rho_;
  this->nUpdates_ = lattice.nUpdates_;
  this->u0_ = lattice.u0_;
  this->links_ = lattice.links_;
  this->randSu3s_ = lattice.randSu3s_;
  this->computeLocalAction = lattice.computeLocalAction;
  this->action_ = lattice.action_;
  this->updateMethod_ = lattice.updateMethod_;
  this->updateFunction_ = lattice.updateFunction_;
  this->parallelFlag_ = lattice.parallelFlag_;
  this->propagatorColumns_ = lattice.propagatorColumns_;
}



Lattice::~Lattice()
{
  // Destructor
}



void Lattice::print()
{
  // Print the links out. A bit redundant due to the interfaces library,
  // but here in case it's needed.
  for (int i = 0; i < this->nLinks_; ++i) {
    cout << this->links_[i] << endl;
  }
}



Matrix3cd& Lattice::getLink(const int link[5])
{
  // Return link specified by index (sanitizes link indices)
  int tempLink[5];
  for (int i = 0; i < 4; ++i)
    tempLink[i] = pyQCD::mod(link[i], this->nEdgePoints);
  tempLink[4] = pyQCD::mod(link[4], 4);

  int index = pyQCD::getLinkIndex(tempLink, this->nEdgePoints);
  
  return this->links_[index];
}



Matrix3cd& Lattice::getLink(const vector<int> link)
{
  // Return link specified by indices
  int tempLink[5];
  for (int i = 0; i < 4; ++i)
    tempLink[i] = pyQCD::mod(link[i], this->nEdgePoints);
  tempLink[4] = pyQCD::mod(link[4], 4);
  
  int index = pyQCD::getLinkIndex(tempLink, this->nEdgePoints);

  return this->links_[index];
}



void Lattice::setLink(const int link[5], const Matrix3cd& matrix)
{
  // Set the value of a link
  this->getLink(link) = matrix;
}



GaugeField Lattice::getSubLattice(const int startIndex, const int size)
{
  // Returns a GaugeField object corresponding to the sub-lattice starting at
  // link index startIndex

  GaugeField out;
  out.resize(size * size * size * size * 4);

  int incrementOne = 4 * this->nEdgePoints;
  int incrementTwo = incrementOne * this->nEdgePoints;
  int incrementThree = incrementTwo * this->nEdgePoints;
  
  int index = 0;

  for (int i = 0; i < size * incrementThree; i += incrementThree) {
    for (int j = 0; j < size * incrementTwo; j += incrementTwo) {
      for (int k = 0; k < size * incrementOne; k += incrementOne) {
	for (int l = 0; l < 4 * size; ++l) {
	  out[index] = this->links_[i + j + k + l];
	  ++index;
	}
      }
    }
  }

  return out;
}



void Lattice::monteCarlo(const int link)
{
  // Iterate through the lattice and update the links using Metropolis
  // algorithm
  // Convert the link index to the lattice coordinates
  int linkCoords[5];
  pyQCD::getLinkIndices(link, this->nEdgePoints, linkCoords);
  // Get the staples
  Matrix3cd staples = (this->*computeStaples)(linkCoords);
  for (int n = 0; n < 10; ++n) {
    // Get a random SU3
    Matrix3cd randSu3 = 
      this->randSu3s_[pyQCD::randomIndex()];
    // Calculate the change in the action
    double actionChange = 
      -this->beta_ / 3.0 *
      ((randSu3 - Matrix3cd::Identity())
       * this->links_[link] 
       * staples).trace().real();
    
    // Was the change favourable? If not, revert the change
    bool isExpMore = exp(-actionChange) >= pyQCD::uni();
    
    if ((actionChange <= 0) || isExpMore)
      this->links_[link] = randSu3 * this->links_[link];
  }
}



void Lattice::monteCarloNoStaples(const int link)
{
  // Iterate through the lattice and update the links using Metropolis
  // algorithm
  // Convert the link index to the lattice coordinates
  int linkCoords[5];
  pyQCD::getLinkIndices(link, this->nEdgePoints, linkCoords);
  // Record the old action contribution
  double oldAction = (this->*computeLocalAction)(linkCoords);
  // Record the old link in case we need it
  Matrix3cd oldLink = this->links_[link];
  
  // Get ourselves a random SU3 matrix for the update
  Matrix3cd randSu3 = 
    this->randSu3s_[pyQCD::randomIndex()];
  // Multiply the site
  this->links_[link] = randSu3 * this->links_[link];
  // What's the change in the action?
  double actionChange = (this->*computeLocalAction)(linkCoords) - oldAction;
  
  // Was the change favourable? If not, revert the change
  bool isExpLess = exp(-actionChange) < pyQCD::uni();
  
  if ((actionChange > 0) && isExpLess)
    this->links_[link] = oldLink;
}



void Lattice::heatbath(const int link)
{
  // Update a single link using heatbath in Gattringer and Lang
  // Convert the link index to the lattice coordinates
  int linkCoords[5];
  pyQCD::getLinkIndices(link, this->nEdgePoints, linkCoords);
  // Calculate the staples matrix A
  Matrix3cd staples = (this->*computeStaples)(linkCoords);
  // Declare the matrix W = U * A
  Matrix3cd W;
  
  // Iterate over the three SU(2) subgroups of W
  for (int n = 0; n < 3; ++n) {
    // W = U * A
    W = this->links_[link]
      * staples;
    double a[4];
    double r[4];
    // Get SU(2) sub-group of W corresponding to index n
    // i.e. n is row/column removed from W to get 2x2 matrix,
    // which is then unitarised. This is matrix V.
    Matrix2cd V = pyQCD::extractSu2(W, a, n);
    
    // Find the determinant needed to unitarise the sub-group
    // of W. a are the coefficients of the Pauli matrices
    // used to generate the SU(2) sub-group matrix
    double a_l = sqrt(abs(a[0] * a[0] +
			  a[1] * a[1] +
			  a[2] * a[2] +
			  a[3] * a[3]));
    
    // X will be the random matrix we generate according to equation
    // 4.45 in Gattringer in Lang (though note that we use a different
    // coefficient here, as otherwise the results come out wrong for
    // some reason.
    Matrix2cd X = this->makeHeatbathSu2(r, a_l);
    // Then calculate the matrix R to update the sub-group and embed it
    // as an SU(3) matrix.
    Matrix3cd R = pyQCD::embedSu2(X * V.adjoint(), n);
    // Do the update
    this->links_[link]
      = R * this->links_[link];
  }
}



void Lattice::update()
{
  // Iterate through the lattice and apply the appropriate update
  // function
  for (int i = 0; i < this->nLinks_; ++i) {
    (this->*updateFunction_)(i);
  }
  this->nUpdates_++;
}



void Lattice::updateSegment(const int startLink, const int nUpdates)
{
  // Updates a segment of the lattice - used for SAP
  // First determine a couple of variables:
  // - How many links do we update? (saves calling of size)
  // - Which link do we start on?
  int nChunkLinks = this->chunkSequence_.size();
  for (int i = 0; i < nUpdates; ++i) {
    for (int j = 0; j < nChunkLinks; ++j) {
      (this->*updateFunction_)(startLink + this->chunkSequence_[j]);
    }
  }
}



void Lattice::runThreads(const int nUpdates, const int remainder)
{
  // Updates every other segment (even or odd, specified by remainder).
  // Algorithm depends on whether the lattice has even or odd dimesnion.
  
  if (remainder == 0) {
    int nChunks = this->evenBlocks_.size();
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < nChunks; ++i) {
      this->updateSegment(this->evenBlocks_[i], nUpdates);
    }
  }
  else {
    int nChunks = this->oddBlocks_.size();
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < nChunks; ++i) {
      this->updateSegment(this->oddBlocks_[i], nUpdates);
    }
  }
}



void Lattice::schwarzUpdate(const int nUpdates)
{
  // Update even and odd blocks using method similar to Schwarz Alternating
  // Procedure.
  this->runThreads(nUpdates, 0);
  this->runThreads(nUpdates, 1);
  this->nUpdates_++;
}



void Lattice::thermalize()
{
  // Update all links until we're at thermal equilibrium
  // Do we do this using OpenMP, or not?
  if (this->parallelFlag_ == 1) {
    while(this->nUpdates_ < 5 * this->nCorrelations)
      // If so, do a Schwarz update thingy (even/odd blocks)
      this->schwarzUpdate(1);
  }
  else {
    while(this->nUpdates_ < 5 * this->nCorrelations)
      this->update();
  }
}



void Lattice::getNextConfig()
{
  // Run nCorrelations updates using parallel or linear method
  if (this->parallelFlag_ == 1) {
    for (int i = 0; i < this->nCorrelations; ++i)
      this->schwarzUpdate(1);
  }
  else {
    for (int i = 0; i < this->nCorrelations; ++i)
      this->update();
  }
  this->reunitarize();
}



void Lattice::reunitarize()
{
  // Do fast polar decomp on all gauge field matrices to ensure they're unitary

#pragma omp parallel for
  for (int i = 0; i < this->nLinks_; ++i) {
    double check = 1.0;
    while (check > 1e-15) {
      Matrix3cd oldMatrix = this->links_[i];
      Matrix3cd inverseMatrix = this->links_[i].inverse();
      double gamma = sqrt(pyQCD::oneNorm(inverseMatrix)
			  / pyQCD::oneNorm(this->links_[i]));
      this->links_[i] *= 0.5 * gamma;
      this->links_[i] += 0.5 / gamma * inverseMatrix.adjoint();
      oldMatrix -= this->links_[i];
      check = sqrt(pyQCD::oneNorm(oldMatrix));
    }
  }
}



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
				  const int nSmears)
{
  // Calculates the loop specified by corners corner1 and corner2 (which must
  // lie in the same plane)
  Matrix3cd out = Matrix3cd::Identity();
  GaugeField linkStore1;
  GaugeField linkStore2;
  // Smear the links if specified, whilst storing the non-smeared ones.
  int timeSliceIncrement = this->nLinks_ / this->nEdgePoints;
  if (nSmears > 0) {
    linkStore1.resize(timeSliceIncrement);
    linkStore2.resize(timeSliceIncrement);
    int corner1Timeslice = pyQCD::mod(corner1[0], this->nEdgePoints);
    int corner2Timeslice = pyQCD::mod(corner2[0], this->nEdgePoints);
    
    for (int i = 0; i < timeSliceIncrement; ++i) {
      linkStore1[i] = this->links_[corner1Timeslice + i];
      linkStore2[i] = this->links_[corner2Timeslice + i];
    }
    this->smearLinks(corner1[0], nSmears);
    this->smearLinks(corner2[0], nSmears);
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
    int corner1Timeslice = pyQCD::mod(corner1[0], this->nEdgePoints);
    int corner2Timeslice = pyQCD::mod(corner2[0], this->nEdgePoints);
    for (int i = 0; i < timeSliceIncrement; ++i) {
      this->links_[corner1Timeslice + i] = linkStore1[i];
      this->links_[corner2Timeslice + i] = linkStore2[i];
    }
  }
  
  return out.trace().real() / 3.0;
}



double Lattice::computeWilsonLoop(const int corner[4], const int r,
				  const int t, const int dimension,
				  const int nSmears)
{
  // Calculates the loop specified by initial corner, width, height and
  // dimension

  GaugeField linkStore1;
  GaugeField linkStore2;
  // Smear the links if specified, whilst storing the non-smeared ones.
  int timeSliceIncrement = this->nLinks_ / this->nEdgePoints;
  if (nSmears > 0) {
    linkStore1.resize(timeSliceIncrement);
    linkStore2.resize(timeSliceIncrement);
    int corner1Timeslice = pyQCD::mod(corner[0], this->nEdgePoints);
    int corner2Timeslice = pyQCD::mod(corner[0] + t, this->nEdgePoints);
    
    for (int i = 0; i < timeSliceIncrement; ++i) {
      linkStore1[i] = this->links_[corner1Timeslice + i];
      linkStore2[i] = this->links_[corner2Timeslice + i];
    }
    this->smearLinks(corner[0], nSmears);
    this->smearLinks(corner[0] + t, nSmears);
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
    int corner1Timeslice = pyQCD::mod(corner[0], this->nEdgePoints);
    int corner2Timeslice = pyQCD::mod(corner[0] + t, this->nEdgePoints);
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
  for (int i = 0; i < this->nEdgePoints; ++i) {
    for (int j = 0; j < this->nEdgePoints; ++j) {
      for (int k = 0; k < this->nEdgePoints; ++k) {
	for (int l = 0; l < this->nEdgePoints; ++l) {
	  for (int m = 0; m < 6; ++m) {
	    int site[4] = {i, j, k, l};
	    Ptot += this->computePlaquette(site,mus[m],nus[m]);
	  }
	}
      }
    }
  }
  // Divide through by number of plaquettes to get mean (simples!)
  return Ptot / (pow(this->nEdgePoints, 4) * 6);
}



double Lattice::computeAverageRectangle()
{
  // Calculate average plaquette operator value
  // mu > nu, so there are six plaquettes at each site.
  int nus[6] = {0, 0, 0, 1, 1, 2};
  int mus[6] = {1, 2, 3, 2, 3, 3};
  double Rtot = 0.0;
  // Pretty simple: step through the matrix and add all plaquettes up
  for (int i = 0; i < this->nEdgePoints; ++i) {
    for (int j = 0; j < this->nEdgePoints; ++j) {
      for (int k = 0; k < this->nEdgePoints; ++k) {
	for (int l = 0; l < this->nEdgePoints; ++l) {
	  for (int m = 0; m < 6; ++m) {
	    int site[4] = {i, j, k, l};
	    Rtot += this->computeRectangle(site, mus[m], nus[m]);
	  }
	}
      }
    }
  }
  // Divide through by number of plaquettes to get mean (simples!)
  return Rtot / (pow(this->nEdgePoints, 4) * 6);
}



double Lattice::computeAverageWilsonLoop(const int r, const int t,
					 const int nSmears)
{
  // Calculates the average of all possible Wilson loops of a given
  // dimension.
  // First off, save the current links and smear all time slices
  GaugeField templinks;
  if (nSmears > 0) {
    templinks = this->links_;
    for (int time = 0; time < this->nEdgePoints; time++) {
      this->smearLinks(time, nSmears);
    }
  }

  double Wtot = 0.0;
  if (this->parallelFlag_ == 1) {
#pragma omp parallel for collapse(5) reduction(+ : Wtot)
    for (int i = 0; i < this->nEdgePoints; ++i) {
      for (int j = 0; j < this->nEdgePoints; ++j) {
	for (int k = 0; k < this->nEdgePoints; ++k) {
	  for (int l = 0; l < this->nEdgePoints; ++l) {
	    for (int m = 1; m < 4; ++m) {
	      int site[4] = {i, j, k, l};
	      // Note, running in parallel causes very
	      // small variations in the final value
	      // of Wtot between consecutive calls
	      // (of the order of 10^-16)
	      Wtot += this->computeWilsonLoop(site, r, t, m, 0);
	    }
	  }
	}
      }
    }
  }
  else {
    for (int i = 0; i < this->nEdgePoints; ++i) {
      for (int j = 0; j < this->nEdgePoints; ++j) {
	for (int k = 0; k < this->nEdgePoints; ++k) {
	  for (int l = 0; l < this->nEdgePoints; ++l) {
	    for (int m = 1; m < 4; ++m) {
	      int site[4] = {i, j, k, l};
	      Wtot += this->computeWilsonLoop(site, r, t, m, 0);
	    }
	  }
	}
      }
    }
  }
  if (nSmears > 0)
    this->links_ = templinks;
  return Wtot / (pow(this->nEdgePoints, 4) * 3);
}



double Lattice::computeMeanLink()
{
  // Pretty simple: step through the matrix and add all link traces up
  double totalLink = 0;
  for (int i = 0; i < this->nLinks_; ++i) {
    totalLink += 1.0 / 3.0
      * this->links_[i].trace().real();
  }
  return totalLink / (4 * pow(this->nEdgePoints, 4));
}



Matrix3cd Lattice::makeRandomSu3()
{
  // Generate a random SU3 matrix, weighted by epsilon
  Matrix3cd A;
  // First generate a random matrix whos elements all lie in/on unit circle  
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A(i, j) = pyQCD::uni();
      A(i, j) *= exp(2  * pyQCD::pi * pyQCD::i * pyQCD::uni());
    }
  }
  // Weight the matrix with weighting eps
  A *= 0.24;
  // Make the matrix traceless and Hermitian
  A(2, 2) = -(A(1, 1) + A(0, 0));
  Matrix3cd B = 0.5 * (A - A.adjoint());
  return B.exp();
}



Matrix2cd Lattice::makeHeatbathSu2(double coefficients[4],
			      const double weighting)
{
  // Generate a random SU2 matrix distributed according to heatbath
  // (See Gattringer and Lang)
  // Initialise lambdaSquared so that we'll go into the for loop
  double lambdaSquared = 2.0;
  // A random squared float to use in the while loop
  double randomSquare = pow(pyQCD::uni(), 2);
  // Loop until lambdaSquared meets the distribution condition
  while (randomSquare > 1.0 - lambdaSquared) {
    // Generate three random floats in (0,1] as per Gattringer and Lang
    double r1 = 1 - pyQCD::uni();
    double r2 = 1 - pyQCD::uni();
    double r3 = 1 - pyQCD::uni();
    // Need a factor of 1.5 here rather that 1/3, not sure why...
    // Possibly due to Nc = 3 in this case
    lambdaSquared = - 1.5 / (weighting * this->beta_) *
      (log(r1) + pow(cos(2 * pyQCD::pi * r2), 2) * log(r3));

    // Get a new random number
    randomSquare = pow(pyQCD::uni(), 2);
  }

  // Get the first of the four elements needed to specify the SU(2)
  // matrix using Pauli matrices
  coefficients[0] = 1 - 2 * lambdaSquared;
  // Magnitude of remaing three-vector is given as follows
  double xMag = sqrt(abs(1 - coefficients[0] * coefficients[0]));

  // Randomize the direction of the remaining three-vector
  // Get a random cos(theta) in [0,1)
  double costheta = -1.0 + 2.0 * pyQCD::uni();
  // And a random phi in [0,2*pi)
  double phi = 2 * pyQCD::pi * pyQCD::uni();

  // We now have everything we need to calculate the remaining three
  // components, so do it
  coefficients[1] = xMag * sqrt(1 - costheta * costheta) * cos(phi);
  coefficients[2] = xMag * sqrt(1 - costheta * costheta) * sin(phi);
  coefficients[3] = xMag * costheta;

  // Now get the SU(2) matrix
  return pyQCD::createSu2(coefficients);
}



Matrix3cd Lattice::computeQ(const int link[5])
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
  
  C *= this->rho_;

  Matrix3cd Omega = C * this->getLink(link).adjoint();
  Matrix3cd OmegaAdjoint = Omega.adjoint() - Omega;
  Matrix3cd out = 0.5 * pyQCD::i * OmegaAdjoint;
  return out - pyQCD::i / 6.0 * OmegaAdjoint.trace() * Matrix3cd::Identity();
}



void Lattice::smearLinks(const int time, const int nSmears)
{
  // Smear the specified time slice by iterating calling this function
  int nSpatialLinks = this->nLinks_ / this->nEdgePoints;

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
	  pyQCD::getLinkIndices(time * nSpatialLinks + j + k, this->nEdgePoints,
				link);
	  Matrix3cd tempMatrix = this->computeQ(link);
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
	  pyQCD::getLinkIndices(time * nSpatialLinks + j + k, this->nEdgePoints,
				link);
	  Matrix3cd tempMatrix = this->computeQ(link);
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



SparseMatrix<complex<double> > Lattice::computeDiracMatrix(const double mass,
							   const double spacing)
{
  // Calculates the Dirac matrix for the current field configuration
  // using Wilson fermions
  
  // Calculate some useful quantities
  int nSites = this->nLinks_ / 4;
  // Create the sparse matrix we're going to return
  SparseMatrix<complex<double> > out(3 * this->nLinks_, 3 * this->nLinks_);

  vector<Tlet> tripletList;
  for (int i = 0; i < 3 * this->nLinks_; ++i) {
    tripletList.push_back(Tlet(i, i, mass + 4 / spacing));
  }
  
  // Now iterate through the matrix and add the neighbouring elements
#pragma omp parallel for
  for (int i = 0; i < this->nLinks_ / 4; ++i) {
    int rowLink[5];
    pyQCD::getLinkIndices(4 * i, this->nEdgePoints, rowLink);

    // We've already calculated the eight neighbours, so we'll deal with those
    // alone
    for (int j = 0; j < 8; ++j) {
      // Get the dimension and index of the current neighbour
      int columnIndex = this->propagatorColumns_[i][j][0];
      int dimension = this->propagatorColumns_[i][j][1];

      // Now we'll get the relevant colour and spin matrices
      Matrix3cd colourMatrix;
      Matrix4cd spinMatrix;
      // (See the action for what's going on here.)
      if (this->propagatorColumns_[i][j][1] > 3) {
	int dimension = this->propagatorColumns_[i][j][1] - 4;
	rowLink[4] = dimension;
	colourMatrix = this->getLink(rowLink);
	spinMatrix = Matrix4cd::Identity() + pyQCD::gammas[dimension];
      }
      else {
	int dimension = this->propagatorColumns_[i][j][1];
	rowLink[dimension]--;
	rowLink[4] = dimension;
	colourMatrix = this->getLink(rowLink).adjoint();
	rowLink[dimension]++;
	spinMatrix = Matrix4cd::Identity() - pyQCD::gammas[dimension];
      }
      // Now loop through the two matrices and put the non-zero entries in the
      // triplet list (basically a tensor product, could put this in a utility
      // function as this might be something that needs to be done again if
      // other actions are implemented).
      for (int k = 0; k < 4; ++k) {
	for (int m = 0; m < 3; ++m) {
	  for (int l = 0; l < 4; ++l) {
	    for (int n = 0; n < 3; ++n) {
	      complex<double> sum = -0.5 / spacing
		* spinMatrix(k, l) * colourMatrix(m, n);
#pragma omp critical
	      if (sum != complex<double>(0,0))
		tripletList.push_back(Tlet(12 * i + 3 * k + m,
					   3 * columnIndex + 3 * l + n, sum));
	    }
	  }
	}
      }
    }
  }
  
  // Add all the triplets to the sparse matrix
  out.setFromTriplets(tripletList.begin(), tripletList.end());
  
  return out;
}



SparseMatrix<complex<double> >
Lattice::computeSmearingOperator(const double smearingParameter,
				 const int nSmears)
{
  // Create the sparse matrix we're going to return
  SparseMatrix<complex<double> > out(3 * this->nLinks_, 3 * this->nLinks_);
  out.setZero();
  // Create the sparse matrix H (eqn 6.40 of Gattringer and Lang)
  SparseMatrix<complex<double> > matrixH(3 * this->nLinks_, 3 * this->nLinks_);
  // This is where we'll store the matrix entries before intialising the matrix
  vector<Tlet> tripletList;

  // Now iterate through the matrix and add the neighbouring elements
#pragma omp parallel for
  for (int i = 0; i < this->nLinks_ / 4; ++i) {
    int rowLink[5];
    pyQCD::getLinkIndices(4 * i, this->nEdgePoints, rowLink);

    // We've already calculated the eight neighbours, so we'll deal with those
    // alone
    for (int j = 0; j < 8; ++j) {
      // Get the dimension and index of the current neighbour
      int columnIndex = this->propagatorColumns_[i][j][0];
      int dimension = this->propagatorColumns_[i][j][1];

      // We're only interested in the spatial links, so skip if the dimension
      // is time
      if (dimension == 0 || dimension == 4)
	break;

      // Now we'll get the relevant colour and spin matrices
      Matrix3cd colourMatrix;
      Matrix4cd spinMatrix = Matrix4cd::Identity();
      // (See the action for what's going on here.)
      if (this->propagatorColumns_[i][j][1] > 3) {
	int dimension = this->propagatorColumns_[i][j][1] - 4;
	rowLink[4] = dimension;
	colourMatrix = this->getLink(rowLink);
      }
      else {
	int dimension = this->propagatorColumns_[i][j][1];
	rowLink[dimension]--;
	rowLink[4] = dimension;
	colourMatrix = this->getLink(rowLink).adjoint();
	rowLink[dimension]++;
      }
      // Now loop through the two matrices and put the non-zero entries in the
      // triplet list (basically a tensor product, could put this in a utility
      // function as this might be something that needs to be done again if
      // other actions are implemented).
      for (int k = 0; k < 4; ++k) {
	for (int m = 0; m < 3; ++m) {
	  for (int l = 0; l < 4; ++l) {
	    for (int n = 0; n < 3; ++n) {
	      complex<double> sum = spinMatrix(k, l) * colourMatrix(m, n);
#pragma omp critical
	      if (sum != complex<double>(0,0))
		tripletList.push_back(Tlet(12 * i + 3 * k + m,
					   3 * columnIndex + 3 * l + n, sum));
	    }
	  }
	}
      }
    }
  }

  // Add all the triplets to the sparse matrix
  matrixH.setFromTriplets(tripletList.begin(), tripletList.end());

  // Need a sparse
  vector<Tlet> identityTripletList;
  for (int i = 0; i < 3 * this->nLinks_; ++i) {
    identityTripletList.push_back(Tlet(i, i, complex<double>(1.0, 0.0)));
  }

  // Create an identity matrix that'll hold the matrix powers in the sum below
  SparseMatrix<complex<double> > matrixHPower(3 * this->nLinks_,
					      3 * this->nLinks_);
  matrixHPower.setFromTriplets(identityTripletList.begin(),
			       identityTripletList.end());

  // Now do the sum as in eqn 6.40 of G+L
  for (int i = 0; i <= nSmears; ++i) {    
    out += pow(smearingParameter, i) * matrixHPower;
    // Need to do the matrix power by hand
    matrixHPower = matrixHPower * matrixH;
  }

  return out;
}



VectorXcd
Lattice::makeSource(const int site[4], const int spin, const int colour,
		    const SparseMatrix<complex<double> >& smearingOperator)
{
  // Generate a (possibly smeared) quark source at the given site, spin and
  // colour
  int nIndices = 3 * this->nLinks_;
  VectorXcd source(nIndices);
  source.setZero(nIndices);

  // Index for the vector point source
  int spatial_index = pyQCD::getLinkIndex(site[0], site[1], site[2], site[3], 0,
					  this->nEdgePoints);
	
  // Set the point source
  int index = colour + 3 * (spin + spatial_index);
  source(index) = 1.0;

  // Now apply the smearing operator
  source = smearingOperator * source;

  return source;
}



vector<MatrixXcd>
Lattice::computePropagator(const double mass, int site[4], const double spacing,
			   const SparseMatrix<complex<double> >& D,
			   const int solverMethod)
{
  // Computes the propagator vectors for the 12 spin-colour indices at
  // the given lattice site, using the Dirac operator

  // How many indices are we dealing with?
  int nSites = this->nLinks_ / 4;
  int nIndices = 3 * this->nLinks_;

  // Index for the vector point source
  int spatial_index = pyQCD::getLinkIndex(site[0], site[1], site[2], site[3], 0,
					  this->nEdgePoints);

  // Declare a variable to hold our propagator
  vector<MatrixXcd> propagator(nSites, MatrixXcd::Zero(12, 12));

  // If using CG, then we need to multiply D by its adjoint
  if (solverMethod == 1) {
    // Get adjoint matrix
    vector<Tlet> tripletList;
    SparseMatrix<complex<double> > Dadj(D.rows(),D.cols());
    for (int i = 0; i < D.outerSize(); ++i)
      for (SparseMatrix<complex<double> >::InnerIterator j(D, i); j; ++j) {
	tripletList.push_back(Tlet(j.col(), j.row(), conj(j.value())));
      }

    Dadj.setFromTriplets(tripletList.begin(), tripletList.end());

    // The matrix we'll be inverting
    SparseMatrix<complex<double> > M = D * Dadj;
    // And the solver
    ConjugateGradient<SparseMatrix<complex<double> > > solver(M);
    solver.setMaxIterations(1000);
    solver.setTolerance(1e-8);

    // Loop through colour and spin indices and invert propagator
    for (int i = 0; i < 4; ++i) {
      for(int j = 0; j < 3; ++j) {
	// Create the source vector
	VectorXcd source(nIndices);
	source.setZero(nIndices);
	
	// Set the point source
	int index = j + 3 * (i + spatial_index);
	source(index) = 1.0;
	
	// Do the inversion
	VectorXcd solution = Dadj * solver.solve(source);
	
	// Add the result to the propagator matrix
	for (int k = 0; k < nSites; ++k) {
	  for (int l = 0; l < 12; ++l) {
	    propagator[k](l, j + 3 * i) = solution(12 * k + l);
	  }
	}
      }
    }
  }
  else {
    // Otherwise just use BiCGSTAB
    BiCGSTAB<SparseMatrix<complex<double> > > solver(D);
    solver.setMaxIterations(1000);
    solver.setTolerance(1e-8);
    
    // Loop through colour and spin indices and invert propagator
    for (int i = 0; i < 4; ++i) {
      for(int j = 0; j < 3; ++j) {
	// Create the source vector
	VectorXcd source(nIndices);
	source.setZero(nIndices);
	
	// Set the point source
	int index = j + 3 * (i + spatial_index);
	source(index) = 1.0;
	
	// Do the inversion
	VectorXcd solution = solver.solve(source);
	
	// Add the result to the propagator matrix
	for (int k = 0; k < nSites; ++k) {
	  for (int l = 0; l < 12; ++l) {
	  propagator[k](l, j + 3 * i) = solution(12 * k + l);
	  }
	}
      }
    }
  }
  
  return propagator;
}



vector<MatrixXcd> Lattice::computePropagator(const double mass, int site[4],
					     const double spacing,
					     const int solverMethod,
					     const int nSmears)
{
  // Computes the propagator vectors for the 12 spin-colour indices at
  // the given lattice site, using the Dirac operator
  // First off, save the current links and smear all time slices
  GaugeField templinks;
  if (nSmears > 0) {
    templinks = this->links_;
    for (int time = 0; time < this->nEdgePoints; time++) {
      this->smearLinks(time, nSmears);
    }
  }
  // Get the dirac matrix
  SparseMatrix<complex<double> > D = this->computeDiracMatrix(mass, spacing);
  // Restore the non-smeared gauge field
  if (nSmears > 0)
    this->links_ = templinks;
  // Calculate and return the propagator
  return this->computePropagator(mass, site, spacing, D, solverMethod);
}



double Lattice::computeLocalWilsonAction(const int link[5])
{
  // Calculate the contribution to the Wilson action from the given link
  int planes[3];
  double Psum = 0.0;

  // Work out which dimension the link is in, since it'll be irrelevant here
  int j = 0;
  for (int i = 0; i < 4; ++i) {
    if (link[4] != i) {
      planes[j] = i;
      ++j;
    }    
  }

  // For each plane, calculate the two plaquettes that share the given link
  for (int i = 0; i < 3; ++i) {
    int site[4] = {link[0], link[1], link[2], link[3]};
    Psum += this->computePlaquette(site, link[4], planes[i]);
    site[planes[i]] -= 1;
    Psum += this->computePlaquette(site, link[4], planes[i]);
  }

  return -this->beta_ * Psum;
}



double Lattice::computeLocalRectangleAction(const int link[5])
{
  // Calculate contribution to improved action from given link

  // First contrbution is from standard Wilson action, so add that in
  double out = 5.0 / 3.0 * this->computeLocalWilsonAction(link);
  double Rsum = 0;

  int planes[3];

  // Work out which dimension the link is in, since it'll be irrelevant here
  int j = 0;
  for (int i = 0; i < 4; ++i) {
    if (link[4] != i) {
      planes[j] = i;
      ++j;
    }
  }
  
  for (int i = 0; i < 3; ++i) {
    int site[4] = {link[0], link[1], link[2], link[3]};
    // Add the six rectangles that contain the link
    Rsum += this->computeRectangle(site, link[4], planes[i]);
    site[link[4]] -= 1;
    Rsum += this->computeRectangle(site, link[4], planes[i]);

    site[link[4]] += 1;
    site[planes[i]] -= 1;
    Rsum += this->computeRectangle(site, link[4], planes[i]);
    site[link[4]] -= 1;
    Rsum += this->computeRectangle(site, link[4], planes[i]);
    
    site[link[4]] += 1;
    site[planes[i]] += 1;
    Rsum += this->computeRectangle(site, planes[i], link[4]);
    site[planes[i]] -= 2;
    Rsum += this->computeRectangle(site, planes[i], link[4]);
  }
  out += this->beta_ / (12 * pow(this->u0_, 2)) * Rsum;
  return out;
}



double Lattice::computeLocalTwistedRectangleAction(const int link[5])
{
  // Calculate contribution to improved action from given link

  // First contrbution is from standard Wilson action, so add that in
  double out = this->computeLocalWilsonAction(link);
  double Tsum = 0;

  int planes[3];

  // Work out which dimension the link is in, since it'll be irrelevant here
  int j = 0;
  for (int i = 0; i < 4; ++i) {
    if (link[4] != i) {
      planes[j] = i;
      ++j;
    }
  }
  
  for (int i = 0; i < 3; ++i) {
    int site[4] = {link[0], link[1], link[2], link[3]};
    // Add the seven twisted rectangles that contain the link
    Tsum += this->computeTwistedRectangle(site, link[4], planes[i]);
    site[link[4]] -= 1;
    Tsum += this->computeTwistedRectangle(site, link[4], planes[i]);

    site[link[4]] += 1;
    site[planes[i]] -= 1;
    Tsum += this->computeTwistedRectangle(site, link[4], planes[i]);
    site[link[4]] -= 1;
    Tsum += this->computeTwistedRectangle(site, link[4], planes[i]);
    
    site[link[4]] += 1;
    site[planes[i]] += 1;
    Tsum += this->computeTwistedRectangle(site, planes[i], link[4]);
    site[planes[i]] -= 1;
    Tsum += this->computeTwistedRectangle(site, planes[i], link[4]);
    site[planes[i]] -= 1;
    Tsum += this->computeTwistedRectangle(site, planes[i], link[4]);
  }
  out -= this->beta_ / (12 * pow(this->u0_, 4)) * Tsum;
  return out;
}



Matrix3cd Lattice::computeWilsonStaples(const int link[5])
{
  // Calculates the sum of staples for the two plaquettes surrounding
  // the link
  int planes[3];

  Matrix3cd out = Matrix3cd::Zero();
  
  // Work out which dimension the link is in, since it'll be irrelevant here
  int j = 0;
  for (int i = 0; i < 4; ++i) {
    if (link[4] != i) {
      planes[j] = i;
      ++j;
    }    
  }

  // For each plane, return the sum of the two link products for the
  // plaquette it resides in
  for (int i = 0; i < 3; ++i) {
    // Create a temporary link array to keep track of which link we're using
    int tempLink[5];
    // Initialise it
    copy(link, link + 5, tempLink);
    
    // First link is U_nu (x + mu)
    tempLink[4] = planes[i];
    tempLink[link[4]] += 1;
    Matrix3cd tempMatrix = this->getLink(tempLink);
    // Next link is U+_mu (x + nu)
    tempLink[4] = link[4];
    tempLink[link[4]] -= 1;
    tempLink[planes[i]] += 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_nu (x)
    tempLink[planes[i]] -= 1;
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // And add it to the output
    out += tempMatrix;

    // First link is U+_nu (x + mu - nu)
    tempLink[link[4]] += 1;
    tempLink[planes[i]] -= 1;
    tempMatrix = this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x - nu)
    tempLink[4] = link[4];
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U_nu (x - nu)
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink);
    // And add it to the output
    out += tempMatrix;
  }
  return out;
}



Matrix3cd Lattice::computeRectangleStaples(const int link[5])
{
  // Calculates the sum of staples for the six rectangles including
  // the link
  int planes[3];
  
  // Work out which dimension the link is in, since it'll be irrelevant here
  int j = 0;
  for (int i = 0; i < 4; ++i) {
    if (link[4] != i) {
      planes[j] = i;
      ++j;
    }    
  }

  Matrix3cd wilsonStaples = this->computeWilsonStaples(link);

  Matrix3cd rectangleStaples = Matrix3cd::Zero();

  // For each plane, return the sum of the two link products for the
  // plaquette it resides in
  for (int i = 0; i < 3; ++i) {
    // Create temporary array to keep track of links
    int tempLink[5];
    // Initialise it
    copy(link, link + 5, tempLink);
    // First link is U_mu (x + mu)
    tempLink[link[4]] += 1;
    Matrix3cd tempMatrix = this->getLink(tempLink);
    // Next link is U_nu (x + 2 * mu)
    tempLink[link[4]] += 1;
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink);
    // Next link U+_mu (x + mu + nu)
    tempLink[link[4]] -= 1;
    tempLink[planes[i]] += 1;
    tempLink[4] = link[4];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x + nu)
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_nu (x)
    tempLink[planes[i]] -= 1;
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Add it to the output
    rectangleStaples += tempMatrix;
    
    // Next is previous rectangle but translated by -1 in current plane
    // First link is U_mu (x + mu)
    tempLink[link[4]] += 1;
    tempLink[4] = link[4];
    tempMatrix = this->getLink(tempLink);
    // Next link is U+_nu (x + 2 * mu - nu)
    tempLink[link[4]] += 1;
    tempLink[planes[i]] -= 1;
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link U+_mu (x + mu - nu)
    tempLink[link[4]] -= 1;
    tempLink[4] = link[4];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x - nu)
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U_nu (x - nu)
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink);
    // Add it to the output
    rectangleStaples += tempMatrix;

    // Next is previous two rectangles but translated by -1 in link axis
    // First link is U_nu (x + mu)
    tempLink[link[4]] += 1;
    tempLink[planes[i]] += 1;
    tempMatrix = this->getLink(tempLink);
    // Next link is U+_mu (x + nu)
    tempLink[planes[i]] += 1;
    tempLink[link[4]] -= 1;
    tempLink[4] = link[4];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x - mu + nu)
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_nu (x - mu)
    tempLink[4] = planes[i];
    tempLink[planes[i]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U_mu (x - mu)
    tempLink[4] = link[4];
    tempMatrix *= this->getLink(tempLink);
    // Add it to the output
    rectangleStaples += tempMatrix;

    // Next is same rectangle but reflected in link axis
    // First link is U+_nu (x + mu - nu)
    tempLink[link[4]] += 2;
    tempLink[planes[i]] -= 1;
    tempLink[4] = planes[i];
    tempMatrix = this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x - nu)
    tempLink[link[4]] -= 1;
    tempLink[4] = link[4];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x - mu - nu)
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U_nu (x - mu - nu)
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink);
    // Next link is U_mu (x - mu)
    tempLink[4] = link[4];
    tempLink[planes[i]] += 1;
    tempMatrix *= this->getLink(tempLink);
    // Add it to the output
    rectangleStaples += tempMatrix;

    // Next we do the rectangles rotated by 90 degrees
    // Link is U_nu (x + mu)
    tempLink[link[4]] += 2;
    tempLink[4] = planes[i];
    tempMatrix = this->getLink(tempLink);
    // Link is U_nu (x + mu + nu)
    tempLink[planes[i]] += 1;
    tempMatrix *= this->getLink(tempLink);
    // Link is U+_mu (x + 2 * nu)
    tempLink[4] = link[4];
    tempLink[link[4]] -= 1;
    tempLink[planes[i]] += 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Link is U+_nu (x + nu)
    tempLink[4] = planes[i];
    tempLink[planes[i]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Link is U+_nu (x)
    tempLink[planes[i]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Add to the sum
    rectangleStaples += tempMatrix;

    // Next flip the previous rectangle across the link axis
    // Link is U+_nu (x + mu - nu)
    tempLink[link[4]] += 1;
    tempLink[planes[i]] -= 1;
    tempLink[4] = planes[i];
    tempMatrix = this->getLink(tempLink).adjoint();
    // Link is U+_nu (x + mu - 2 * nu)
    tempLink[planes[i]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Link is U+_mu (x - 2 * nu)
    tempLink[4] = link[4];
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Link is U_nu (x - 2 * nu)
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink);
    // Link is U_nu (x - nu)
    tempLink[planes[i]] += 1;
    tempMatrix *= this->getLink(tempLink);
    // Add to the sum
    rectangleStaples += tempMatrix;
  }

  return 5.0 / 3.0 * wilsonStaples 
    - rectangleStaples / (12.0 * pow(this->u0_, 2));
}



Matrix3cd Lattice::computeTwistedRectangleStaples(const int link[5])
{
  cout << "Error! Cannot compute sum of staples for twisted rectangle "
       << "operator." << endl;
  return Matrix3cd::Identity();
}
