#include <lattice.hpp>
#include <pyQCD_utils.hpp>

Lattice::Lattice(const int nEdgePoints, const double beta, const double u0,
		 const int action, const int nCorrelations, const double rho,
		 const double epsilon, const int updateMethod,
		 const int parallelFlag)
{
  // Default constructor. Assigns function arguments to member variables
  // and initializes links.
  this->nEdgePoints = nEdgePoints;
  this->nLinks_ = int(pow(this->nEdgePoints, 4) * 4);
  this->beta_ = beta;
  this->nCorrelations = nCorrelations;
  this->epsilon_ = epsilon;
  this->rho_ = rho;
  this->nUpdates_ = 0;
  this->u0_ = u0;
  this->action_ = action;
  this->updateMethod_ = updateMethod;
  this->parallelFlag_ = parallelFlag;

  // Initialize parallel Eigen
  initParallel();

  // Resize the link vector and assign each link a random SU3 matrix
  // Also set up the linkIndices vector
  this->linkIndices_.resize(this->nLinks_);
  this->links_.resize(this->nLinks_);

  for (int i = 0; i < this->nLinks_; ++i) {
    this->links_[i] = this->makeRandomSu3();
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
	    int index = m + this->nEdgePoints
	      * (l + this->nEdgePoints
		 * (k + this->nEdgePoints
		    * (j + this->nEdgePoints * i)));
	    this->chunkSequence_.push_back(index);
	  }
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
  this->epsilon_ = lattice.epsilon_;
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
}



Lattice::~Lattice()
{
  // Destructor
}



void convertIndex(const int index, int link[5])
{
  // Converts single link index to set of link coordinates
  int tempIndex = index;
  link[4] = tempIndex % 4;
  for (int i = 3; i > -1; ++i) {
    tempIndex /= this->nEdgePoints;
    link[i] = tempIndex % this->nEdgePoints;
  }
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
  tempLink[4] = pyQCD::mod(link[4], this->nEdgePoints);

  int index = tempLink[4] + 
    this->nEdgePoints * (tempLink[3] + this->nEdgePoints
			 * (tempLink[2] + this->nEdgePoints
			    * (tempLink[1] + this->nEdgePoints
			       * tempLink[0])));

  return this->links_[index];
}



Matrix3cd& Lattice::getLink(const vector<int> link)
{
  // Return link specified by indices
  int tempLink[5];
  for (int i = 0; i < 5; ++i)
    tempLink[i] = pyQCD::mod(link[i], this->nEdgePoints);
  tempLink[4] = pyQCD::mod(link[4], this->nEdgePoints);
  
  int index = tempLink[4] + 
    this->nEdgePoints * (tempLink[3] + this->nEdgePoints
			 * (tempLink[2] + this->nEdgePoints
			    * (tempLink[1] + this->nEdgePoints
			       * tempLink[0])));

  return this->links_[index];
}



void Lattice::monteCarlo(const int link)
{
  // Iterate through the lattice and update the links using Metropolis
  // algorithm
  // Convert the link index to the lattice coordinates
  int linkCoords[5];
  this->convertIndex(link, linkCoords);
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
  this->convertIndex(link, linkCoords);
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



void Lattice::heatbath(const int link[5])
{
  // Update a single link using heatbath in Gattringer and Lang
  // Calculate the staples matrix A
  Matrix3cd staples = (this->*computeStaples)(link);
  // Declare the matrix W = U * A
  Matrix3cd W;
  
  // Iterate over the three SU(2) subgroups of W
  for (int n = 0; n < 3; ++n) {
    // W = U * A
    W = this->links_[link[0]][link[1]][link[2]][link[3]][link[4]]
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
    this->links_[link[0]][link[1]][link[2]][link[3]][link[4]]
      = R * this->links_[link[0]][link[1]][link[2]][link[3]][link[4]];
  }
}



void Lattice::update()
{
  // Iterate through the lattice and apply the appropriate update
  // function
  int nLinks = this->linkIndices_.size();
  for (int i = 0; i < nLinks; ++i) {
    int link[5];
    copy(this->linkIndices_[i].begin(),
	 this->linkIndices_[i].end(),
	 link);
    (this->*updateFunction_)(link);
  }
  this->nUpdates_++;
}



void Lattice::updateSegment(const int n0, const int n1, const int n2,
			    const int n3, const int chunkSize,
			    const int nUpdates)
{
  // Updates a segment of the lattice - used for SAP
  for (int i = 0; i < nUpdates; ++i) {
    for (int j = n0; j < n0 + chunkSize; ++j) {
      for (int k = n1; k < n1 + chunkSize; ++k) {
	for (int l = n2; l < n2 + chunkSize; ++l) {
	  for (int m = n3; m < n3 + chunkSize; ++m) {
	    for (int n = 0; n < 4; ++n) {
	      // We'll need an array with the link indices
	      int link[5] = {j, k, l, m, n};
	      (this->*updateFunction_)(link);
	    }
	  }
	}
      }
    }
  }
}



void Lattice::runThreads(const int chunkSize, const int nUpdates,
			 const int remainder)
{
  // Updates every other segment (even or odd, specified by remainder).
  // Algorithm depends on whether the lattice has even or odd dimesnion.
#pragma omp parallel for schedule(static, 1) collapse(4)
  for (int i = 0; i < this->nEdgePoints; i += chunkSize) {
    for (int j = 0; j < this->nEdgePoints; j += chunkSize) {
      for (int k = 0; k < this->nEdgePoints; k += chunkSize) {
	for (int l = 0; l < this->nEdgePoints; l += chunkSize) {
	  if (((i + j + k + l) / chunkSize) % 2 == remainder) {
	    this->updateSegment(i, j, k, l, chunkSize, nUpdates);
	  }
	}
      }
    }
  }
}



void Lattice::schwarzUpdate(const int chunkSize, const int nUpdates)
{
  // Update even and odd blocks using method similar to Schwarz Alternating
  // Procedure.
  this->runThreads(chunkSize, nUpdates, 0);
  this->runThreads(chunkSize, nUpdates, 1);
  this->nUpdates_++;
}

void Lattice::thermalize()
{
  // Update all links until we're at thermal equilibrium
  // Do we do this using OpenMP, or not?
  if (this->parallelFlag_ == 1) {
    while(this->nUpdates_ < 5 * this->nCorrelations)
      // If so, do a Schwarz update thingy (even/odd blocks)
      this->schwarzUpdate(4,1);
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
      this->schwarzUpdate(4,1);
  }
  else {
    for (int i = 0; i < this->nCorrelations; ++i)
      this->update();
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
  SubField linkStore1;
  SubField linkStore2;
  // Smear the links if specified, whilst storing the non-smeared ones.
  if (nSmears > 0) {
    linkStore1 = this->links_[pyQCD::mod(corner1[0], this->nEdgePoints)];
    linkStore2 = this->links_[pyQCD::mod(corner2[0], this->nEdgePoints)];
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
    this->links_[pyQCD::mod(corner1[0], this->nEdgePoints)] = linkStore1;
    this->links_[pyQCD::mod(corner2[0], this->nEdgePoints)] = linkStore2;
  }
  
  return out.trace().real() / 3.0;
}



double Lattice::computeWilsonLoop(const int corner[4], const int r,
				  const int t, const int dimension,
				  const int nSmears)
{
  // Calculates the loop specified by initial corner, width, height and
  // dimension

  SubField linkStore1;
  SubField linkStore2;
  // Smear the links if specified, whilst storing the non-smeared ones.
  if (nSmears > 0) {
    linkStore1 = this->links_[pyQCD::mod(corner[0], this->nEdgePoints)];
    linkStore2 = this->links_[pyQCD::mod(corner[0] + t, this->nEdgePoints)];
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
    out *= this->getLink(link);
    link[dimension]--;
  }

  link[4] = 0;

  // Calculate the second temporal edge
  for (int i = 0; i < t; ++i) {
    out *= this->getLink(link);
    link[0]--;
  }

  // Restore the old links
  if (nSmears > 0) {
    this->links_[pyQCD::mod(corner[0], this->nEdgePoints)] = linkStore1;
    this->links_[pyQCD::mod(corner[0] + t, this->nEdgePoints)] = linkStore2;
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
  GaugeField templinks = this->links_;
  for (int time = 0; time < this->nEdgePoints; time++) {
    this->smearLinks(time, nSmears);
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
  this->links_ = templinks;
  return Wtot / (pow(this->nEdgePoints, 4) * 3);
}



double Lattice::computeMeanLink()
{
  // Pretty simple: step through the matrix and add all link traces up
  double totalLink = 0;
  for (int i = 0; i < this->nEdgePoints; ++i) {
    for (int j = 0; j < this->nEdgePoints; ++j) {
      for (int k = 0; k < this->nEdgePoints; ++k) {
	for (int l = 0; l < this->nEdgePoints; ++l) {
	  for (int m = 0; m < 4; ++m) {
	    totalLink += 1.0 / 3.0
	      * this->links_[i][j][k][l][m].trace().real();
	  }
	}
      }
    }
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
  A *= this->epsilon_;
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
  if (this->parallelFlag_ == 1) {

    for (int i = 0; i < nSmears; ++i) {
      // Iterate through all the links and calculate the new ones from
      // the existing ones.    
      SubField newLinks(this->nEdgePoints, 
			Sub2Field(this->nEdgePoints,
				  Sub3Field(this->nEdgePoints, 
					    Sub4Field(4))));
#pragma omp parallel for collapse(3)
      for (int j = 0; j < this->nEdgePoints; ++j) {
	for (int k = 0; k < this->nEdgePoints; ++k) {
	  for (int l = 0; l < this->nEdgePoints; ++l) {
	    // NB, spatial links only, so l > 0!
	    newLinks[j][k][l][0] =
	      this->links_[pyQCD::mod(time, this->nEdgePoints)][j][k][l][0];
	    for (int m = 1; m < 4; ++m) {
	      // Create a temporary matrix to store the new link
	      int link[5] = {time, j, k, l, m};
	      Matrix3cd tempMatrix = this->computeQ(link);
	      newLinks[j][k][l][m] = (pyQCD::i * tempMatrix).exp()
		* this->getLink(link);
	    }
	  }
	}
      }
      // Apply the changes to the existing lattice.
      this->links_[pyQCD::mod(time, this->nEdgePoints)] = newLinks;
    }
  }
  else {

    for (int i = 0; i < nSmears; ++i) {
      // Iterate through all the links and calculate the new ones from
      // the existing ones.    
      SubField newLinks(this->nEdgePoints, 
			Sub2Field(this->nEdgePoints,
				  Sub3Field(this->nEdgePoints, 
					    Sub4Field(4))));

      for (int j = 0; j < this->nEdgePoints; ++j) {
	for (int k = 0; k < this->nEdgePoints; ++k) {
	  for (int l = 0; l < this->nEdgePoints; ++l) {
	    // NB, spatial links only, so l > 0!
	    newLinks[j][k][l][0] =
	      this->links_[pyQCD::mod(time, this->nEdgePoints)][j][k][l][0];
	    for (int m = 1; m < 4; ++m) {
	      // Create a temporary matrix to store the new link
	      int link[5] = {time, j, k, l, m};
	      Matrix3cd tempMatrix = this->computeQ(link);
	      newLinks[j][k][l][m] = (pyQCD::i * tempMatrix).exp()
		* this->getLink(link);
	    }
	  }
	}
      }
      // Apply the changes to the existing lattice.
      this->links_[pyQCD::mod(time, this->nEdgePoints)] = newLinks;
    }
  }
}



SparseMatrix<complex<double> > Lattice::computeDiracMatrix(const double mass,
							   const double spacing)
{
  // Calculates the Dirac matrix for the current field configuration
  // using Wilson fermions

  // TODO - pass the SparseMatrix in by reference to save computing time
  
  // Calculate some useful quantities
  int nIndices = int(12 * pow(this->nEdgePoints, 4));
  int nSites = int(pow(this->nEdgePoints, 4));
  // Create the sparse matrix we're going to return
  SparseMatrix<complex<double> > out(nIndices, nIndices);

  vector<Tlet> tripletList;
  for (int i = 0; i < nIndices; ++i) {
    tripletList.push_back(Tlet(i, i, mass + 4 / spacing));
  }

  // Create and initialise a vector of the space, lorentz and colour indices
  vector<vector<int> > indices(pow(this->nEdgePoints, 4) * 12, 
			       vector<int>(6));
  int index = 0;
  for (int i = 0; i < this->nEdgePoints; ++i) {
    for (int j = 0; j < this->nEdgePoints; ++j) {
      for (int k = 0; k < this->nEdgePoints; ++k) {
	for (int l = 0; l < this->nEdgePoints; ++l) {
	  for (int alpha = 0; alpha < 4; ++alpha) {
	    for (int a = 0; a < 3; a++) {
	      indices[index][0] = i;
	      indices[index][1] = j;
	      indices[index][2] = k;
	      indices[index][3] = l;
	      indices[index][4] = alpha;
	      indices[index][5] = a;
	      index++;
	    }
	  }
	}
      }
    }
  }
  
  // Now iterate through the matrix and add the various elements to the
  // vector of triplets
#pragma omp parallel for
  for (int i = 0; i < nIndices; ++i) {
    int siteI[4] = {indices[i][0],
		    indices[i][1],
		    indices[i][2],
		    indices[i][3]};
    
    for (int j = 0; j < nIndices; ++j) {
      int m = i / 12;
      int n = j / 12;

      // We can determine whether the spatial indices are going
      // to trigger the delta function in advance, and hence
      // if that's not going to happen we can save ourself a lot
      // of hassle
      bool isAdjacent = false;
      for (int k = 0; k < 4; ++k) {
	// Store this to save calculating it twice
	int nOff = pow(this->nEdgePoints, k);

	bool isJustBelow = m == pyQCD::mod(n + nOff, nSites);
	bool isJustAbove = m == pyQCD::mod(n - nOff, nSites);
	// Are the two sites adjacent to one another?
	if (isJustBelow || isJustAbove) {
	  isAdjacent = true;
	  break;
	}
      }
      // If the two sites are adjacent, then there is some
      // point in doing the sum
      if (isAdjacent) {
	// First we'll need something to put the sum into
	complex<double> sum = complex<double>(0.0, 0.0);	
	// First create an array for the site specified by the index i	
	int siteJ[4] = {indices[j][0],
			indices[j][1],
			indices[j][2],
			indices[j][3]};
	for (int k = 0; k < 4; ++k) {
	  // First need to implement the kronecker delta in the sum of mus,
	  // which'll be horrendous, but hey...

	  // Add (or subtract) the corresponding mu vector from the second
	  // lattice site
	  siteJ[k] = pyQCD::mod(siteJ[k] + 1,
				this->nEdgePoints);
	
	  // If they are, then we have ourselves a matrix element
	  // First test for when mu is positive, as then we'll need to deal
	  // with the +ve or -ve cases slightly differently
	  if (equal(siteI, siteI + 4, siteJ)) {
	    // Create and intialise the link we'll be using
	    int link[5];
	    copy(siteI, siteI + 4, link);
	    link[4] = k;
	    // Then we'll need a colour matrix given by the link
	    Matrix3cd U;
	    // And get the gamma matrix (1 - gamma) in the sum
	    Matrix4cd lorentz = 
	      Matrix4cd::Identity() - pyQCD::gammas[k];
	    // So, if the current mu is positive, just get
	    // the plain old link given by link as normal
	    U = this->getLink(link);
	    // Mutliply the matrix elements together and add
	    // them to the sum.
	    sum += lorentz(indices[i][4], indices[j][4]) 
	      * U(indices[i][5], indices[j][5]);
	  }
	  
	  siteJ[k] = pyQCD::mod(siteJ[k] - 2,
				this->nEdgePoints);

	  if (equal(siteI, siteI + 4, siteJ)) {
	    // Create and intialise the link we'll be using
	    int link[5];
	    copy(siteI, siteI + 4, link);
	    link[4] = k;
	    // Then we'll need a colour matrix given by the link
	    Matrix3cd U;
	    // And get the gamma matrix (1 - gamma) in the sum
	    Matrix4cd lorentz = 
	      Matrix4cd::Identity() + pyQCD::gammas[k];
	    // So, if the current mu is positive, just get
	    // the plain old link given by link as normal
	    link[k] -= 1;
	    U = this->getLink(link).adjoint();
	    // Mutliply the matrix elements together and add
	    // them to the sum.
	    sum += lorentz(indices[i][4], indices[j][4]) 
	      * U(indices[i][5], indices[j][5]);
	  }
	}
	// Divide the sum through by -2 * spacing
	sum /= -(2.0 * spacing);
	// Make sure OpemMP doesn't conflict with itself
#pragma omp critical
	if (sum.imag() != 0.0 && sum.real() != 0.0)
	  // Add the sum to the list of triplets
	  tripletList.push_back(Tlet(i, j, sum));
      }
      else {
	// If the sites aren't neighbours, skip ahead to the next
	// site, as there's no point doing it for the other colours
	// and spin indices.
	j = (n + 1) * 12 - 1;
      }
    }
  }
  
  // Add all the triplets to the sparse matrix
  out.setFromTriplets(tripletList.begin(), tripletList.end());
  
  return out;
}



MatrixXcd Lattice::computePropagator(const double mass, int site[4],
				     const double spacing,
				     const SparseMatrix<complex<double> >& D)
{
  // Computes the propagator vectors for the 12 spin-colour indices at
  // the given lattice site, using the Dirac operator

  // How many indices are we dealing with?
  int nSites = int(pow(this->nEdgePoints, 4));
  int nIndices = 12 * nSites;
  // Declare our solver
  BiCGSTAB<SparseMatrix<complex<double> > > solver(D);

  // Declare a variable to hold our propagator
  MatrixXcd propagator(12, 12);

  // Loop through colour and spin indices and invert propagator
  for (int i = 0; i < 4; ++i) {
    for(int j = 0; j < 3; ++j) {
      // Create the source vector
      VectorXcd source(nIndices);
      source.setZero(nIndices);
  
      // Set the point source
      int spatial_index = site[3] + this->nEdgePoints 
	* (site[2] + this->nEdgePoints 
	   * (site[1] + this->nEdgePoints * site[0]));
      int index = j + 3 * (i + 4 * spatial_index);
      source(index) = 1.0;
      
      // Do the inversion
      VectorXcd solution = solver.solve(source);

      // Add the result to the propagator matrix
      for (int k = 0; k < 12; ++k) {
	propagator(k, j + 3 * i) = solution(12 * spatial_index + k);
      }
    }
  }
  
  return propagator;
}



MatrixXcd Lattice::computePropagator(const double mass, int site[4],
				     const double spacing)
{
  // Computes the propagator vectors for the 12 spin-colour indices at
  // the given lattice site, using the Dirac operator
  SparseMatrix<complex<double> > D = this->computeDiracMatrix(mass, spacing);
  
  return this->computePropagator(mass, site, spacing, D);
}



MatrixXcd Lattice::computeZeroMomPropagator(const double mass, const int time,
					    const double spacing)
{
  // Computes the projected zero momentum propagator

  SparseMatrix<complex<double> > D = this->computeDiracMatrix(mass, spacing);

  int nSpatialIndices = int(pow(this->nEdgePoints, 3));
  int nIndices = int(12 * nSpatialIndices);
  SparseMatrix<complex<double> > source(int(this->nEdgePoints * nIndices),
					nIndices);

  vector<Tlet> tripletList;
  for (int i = 0; i < nIndices; ++i) {
    tripletList.push_back(Tlet(time * nIndices + i, i, 1.0));
  }
  source.setFromTriplets(tripletList.begin(), tripletList.end());

  BiCGSTAB<SparseMatrix<complex<double> > > solver(D);
  SparseMatrix<complex<double> > propagators = solver.solve(source);

  MatrixXcd sum = Matrix<complex<double>, 12, 12>::Zero();

  for (int i = 0; i < nSpatialIndices; ++i) {
    for (int j = 0; j < 12; ++j) {
      for (int k = 0; k < 12; ++k) {
	sum(j, k) = propagators.coeffRef(12 * (nSpatialIndices * time + i) + j,
					 12 * i + k);
      }
    }
  }

  return sum / pow(this->nEdgePoints, 3);
}



vector<MatrixXcd> Lattice::computePropagators(const double mass,
					      const double spacing)
{
  // Computes all propagators at all lattice sites

  // Determine the number spatial sites
  int nSites = int(pow(this->nEdgePoints, 4));

  // Create the Direc operator
  SparseMatrix<complex<double> > D = this->computeDiracMatrix(mass, spacing);
  
  // Declare the output
  vector<MatrixXcd> propagators(nSites);

  for (int i = 0; i < this->nEdgePoints; ++i) {
    for (int j = 0; j < this->nEdgePoints; ++j) {
      for (int k = 0; k < this->nEdgePoints; ++k) {
	for (int l = 0; l < this->nEdgePoints; ++l) {
	  int index = l + this->nEdgePoints 
	    * (k + this->nEdgePoints * (j + this->nEdgePoints * i));
	  int site[4] = {i, j, k, l};
	  propagators[index] = this->computePropagator(mass, site, spacing, D);
	}
      }
    }
  }

  return propagators;
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
