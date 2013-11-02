#include <lattice.hpp>
#include <utils.hpp>

Lattice::Lattice(const int spatialExtent, const int temporalExtent,
		 const double beta, const double u0, const int action,
		 const int nCorrelations, const int updateMethod,
		 const int parallelFlag, const int chunkSize,
		 const int randSeed)
{
  // Default constructor. Assigns function arguments to member variables
  // and initializes links.
  this->spatialExtent = spatialExtent;
  this->temporalExtent = temporalExtent;
  this->nLinks_ = int(pow(this->spatialExtent, 3) * this->temporalExtent * 4);
  this->beta_ = beta;
  this->nCorrelations = nCorrelations;
  this->nUpdates_ = 0;
  this->u0_ = u0;
  this->action_ = action;
  this->updateMethod_ = updateMethod;
  this->parallelFlag_ = parallelFlag;
  
  if (randSeed > -1)
    this->rng.setSeed(randSeed);

  // Initialize parallel Eigen
  initParallel();

  // Resize the link vector and assign each link a random SU3 matrix
  // Also set up the propagatorColumns vector
  this->links_.resize(this->nLinks_);
  this->propagatorColumns_
    = vector<vector<vector<int> > >(this->nLinks_ / 4,
				    vector<vector<int> >(8,vector<int>(3,0)));

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
#pragma omp parallel for
  for (int i = 0; i < this->nLinks_; i += 4) {
    
    // Initialize the relevant site vector for the row index
    int rowLink[5];
    pyQCD::getLinkIndices(i, this->spatialExtent, this->temporalExtent, rowLink);
    
    
    // Loop through the offsets
    for (int j = 0; j < 8; ++j) {
      // Get the coordinates for the column
      int columnLink[5] = {0, 0, 0, 0, 0};
      int boundaryCondition = 1;
      
      // Loop through the coordiates for the pair of sites and see if they're
      // neighbours
      int latticeSize[4] = {this->temporalExtent,
			    this->spatialExtent,
			    this->spatialExtent,
			    this->spatialExtent};
      for (int k = 0; k < 4; ++k) {
	columnLink[k] = pyQCD::mod(rowLink[k] + offsets[j][k],
				   latticeSize[k]);
      }

      // Check whether antiperiodic boundary conditions need to be applied
      if (rowLink[0] + offsets[j][0] >= latticeSize[0] ||
	  rowLink[0] + offsets[j][0] < 0)
	// Apply antiperiodic boundary conditions
	boundaryCondition = -1;

      int columnIndex = pyQCD::getLinkIndex(columnLink, this->spatialExtent);
      
      this->propagatorColumns_[i / 4][j][0] = columnIndex;
      this->propagatorColumns_[i / 4][j][1] = j;
      this->propagatorColumns_[i / 4][j][2] = boundaryCondition;
      
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
      this->updateFunction_ = &Lattice::metropolisNoStaples;
    }
  }
  else if (updateMethod == 1) {
    if (action != 2) {
      this->updateFunction_ = &Lattice::metropolis;
    }
    else {
      cout << "Warning! Heatbath updates are not compatible with twisted "
	   << "rectangle action. Using Monte Carlo instead" << endl;
      this->updateFunction_ = &Lattice::metropolisNoStaples;
    }
  }
  else if (updateMethod == 2) {
    this->updateFunction_ = &Lattice::metropolisNoStaples;
  }
  else {
    cout << "Warning! Specified update method does not exist!" << endl;
    if (action != 2) {
      this->updateFunction_ = &Lattice::heatbath;
    }
    else {
      cout << "Warning! Heatbath updates are not compatible with twisted "
	   << "rectangle action. Using Monte Carlo instead" << endl;
      this->updateFunction_ = &Lattice::metropolisNoStaples;
    }
  }
  
  // Initialize series of offsets used when doing block updates
  
  for (int i = 0; i < chunkSize; ++i) {
    for (int j = 0; j < chunkSize; ++j) {
      for (int k = 0; k < chunkSize; ++k) {
	for (int l = 0; l < chunkSize; ++l) {
	  for (int m = 0; m < 4; ++m) {
	    // We'll need an array with the link indices
	    int index = pyQCD::getLinkIndex(i, j, k, l, m, this->spatialExtent);
	    this->chunkSequence_.push_back(index);
	  }
	}
      }
    }
  }
  
  for (int i = 0; i < this->temporalExtent; i += chunkSize) {
    for (int j = 0; j < this->spatialExtent; j += chunkSize) {
      for (int k = 0; k < this->spatialExtent; k += chunkSize) {
	for (int l = 0; l < this->spatialExtent; l += chunkSize) {
	  // We'll need an array with the link indices
	  int index = pyQCD::getLinkIndex(i, j, k, l, 0, this->spatialExtent);
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
  this->spatialExtent = lattice.spatialExtent;
  this->temporalExtent = lattice.temporalExtent;
  this->nLinks_ = lattice.nLinks_;
  this->beta_ = lattice.beta_;
  this->nCorrelations = lattice.nCorrelations;
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
  this->randSeed_ = lattice.randSeed_;
  if (this->randSeed_ > -1)
    this->rng.setSeed(this->randSeed_);
}



Lattice& Lattice::operator=(const Lattice& lattice)
{
  // Default constructor. Assigns function arguments to member variables
  // and initializes links.
  this->spatialExtent = lattice.spatialExtent;
  this->temporalExtent = lattice.temporalExtent;
  this->nLinks_ = lattice.nLinks_;
  this->beta_ = lattice.beta_;
  this->nCorrelations = lattice.nCorrelations;
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
  this->randSeed_ = lattice.randSeed_;
  if (this->randSeed_ > -1)
    this->rng.setSeed(this->randSeed_);

  return *this;
}



Lattice::~Lattice()
{
  // Destructor
}
