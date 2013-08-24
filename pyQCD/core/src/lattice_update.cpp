#include <lattice.hpp>
#include <utils.hpp>

void Lattice::monteCarlo(const int link)
{
  // Iterate through the lattice and update the links using Metropolis
  // algorithm
  // Convert the link index to the lattice coordinates
  int linkCoords[5];
  pyQCD::getLinkIndices(link, this->spatialExtent, this->temporalExtent,
			linkCoords);
  // Get the staples
  Matrix3cd staples = (this->*computeStaples)(linkCoords);
  for (int n = 0; n < 10; ++n) {
    // Get a random SU3
    Matrix3cd randSu3 = 
      this->randSu3s_[rng.generateInt()];
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
  pyQCD::getLinkIndices(link, this->spatialExtent, this->temporalExtent,
			linkCoords);
  // Record the old action contribution
  double oldAction = (this->*computeLocalAction)(linkCoords);
  // Record the old link in case we need it
  Matrix3cd oldLink = this->links_[link];
  
  // Get ourselves a random SU3 matrix for the update
  Matrix3cd randSu3 = 
    this->randSu3s_[rng.generateInt()];
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
  pyQCD::getLinkIndices(link, this->spatialExtent, this->temporalExtent,
			linkCoords);
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
