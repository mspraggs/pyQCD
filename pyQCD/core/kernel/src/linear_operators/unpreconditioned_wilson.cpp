#include <linear_operators/unpreconditioned_wilson.hpp>

UnpreconditionedWilson::UnpreconditionedWilson(
    const double mass, const vector<complex<double> >& boundaryConditions,
    Lattice* lattice) : lattice_(lattice)
{
  // Class constructor - we set the fermion mass, create a pointer to the 
  // lattice and compute the frequently used spin structures used within the
  // Dirac operator.
  this->mass_ = mass;
  this->operatorSize_ 
    = 12 * int(pow(lattice->spatialExtent, 3)) * lattice->temporalExtent;

  for (int i = 0; i < 4; ++i) {
    this->spinStructures_.push_back(Matrix4cd::Identity() - pyQCD::gammas[i]);
  }

  for (int i = 0; i < 4; ++i) {
    this->spinStructures_.push_back(Matrix4cd::Identity() + pyQCD::gammas[i]);
  }
  // Here are some Hermitian gamma matrices to use when computing a Hermitian
  // version of the gamma operator
  for (int i = 0; i < 8; ++i) {
    this->hermitianSpinStructures_.push_back(this->spinStructures_[i] 
					     * pyQCD::gamma5);
  }

  int latticeShape[4] = {this->lattice_->temporalExtent,
			 this->lattice_->spatialExtent,
			 this->lattice_->spatialExtent,
			 this->lattice_->spatialExtent};

  for (int i = 0; i < this->operatorSize_ / 12; ++i) {

    vector<int> neighbours(8, 0);
    vector<complex<double> > siteBoundaryConditions(8, complex<double>(1.0,
								       0.0));

    // Determine the coordinates of the site we're on
    int site[5]; // The coordinates of the lattice site
    pyQCD::getLinkIndices(4 * i, this->lattice_->spatialExtent,
			  this->lattice_->temporalExtent, site);

    int siteBehind[5]; // Site/link index for the site/link behind us
    int siteAhead[5]; // Site/link index for the site/link in front of us

    // Loop over the four gamma indices (mu) in the sum inside the Wilson action
    for (int mu = 0; mu < 4; ++mu) {

      // Determine whether we need to apply boundary conditions

      copy(site, site + 5, siteBehind);
      if (site[mu] - 1 < 0 || site[mu] - 1 >= latticeShape[mu])
	siteBoundaryConditions[mu] = boundaryConditions[mu % 4];

      if (site[mu] + 1 < 0 || site[mu] + 1 >= latticeShape[mu])
	siteBoundaryConditions[mu + 4] = boundaryConditions[mu % 4];
          
      // Now we determine the indices of the neighbouring links

      siteBehind[mu] = pyQCD::mod(siteBehind[mu] - 1, latticeShape[mu]);
      int siteBehindIndex = pyQCD::getLinkIndex(siteBehind,
						 latticeShape[1]) / 4;

      copy(site, site + 5, siteAhead);
      siteAhead[mu] = pyQCD::mod(siteAhead[mu] + 1, latticeShape[mu]);
      int siteAheadIndex = pyQCD::getLinkIndex(siteAhead,
						latticeShape[1]) / 4;

      neighbours[mu] = siteBehindIndex;
      neighbours[mu + 4] = siteAheadIndex;
    }
    this->boundaryConditions_.push_back(siteBoundaryConditions);
    this->nearestNeighbours_.push_back(neighbours);
  }
}



UnpreconditionedWilson::~UnpreconditionedWilson()
{
  // Empty destructor
}



VectorXcd UnpreconditionedWilson::apply(const VectorXcd& psi)
{
  // Right multiply a vector by the operator
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

#pragma omp parallel for
  for (int i = 0; i < this->operatorSize_; ++i) {
    int etaSiteIndex = i / 12; // Site index of the current row in 
    int alpha = (i % 12) / 3; // Spin index of the current row in eta
    int a = i % 3; // Colour index of the current row in eta

    // Now add the mass term
    eta(i) = (1 + 3 / this->lattice_->chi() + this->mass_) * psi(i);

    // Loop over the four gamma indices (mu) in the sum inside the Wilson action
    for (int mu = 0; mu < 4; ++mu) {
      
      // Now we determine the indices of the neighbouring links

      int siteBehindIndex = this->nearestNeighbours_[etaSiteIndex][mu];
      int siteAheadIndex = this->nearestNeighbours_[etaSiteIndex][mu + 4];

      // Now loop over spin and colour indices for the portion of the row we're
      // on

      for (int j = 0; j < 12; ++j) {
	int beta = j / 3; // Compute spin
	int b = j % 3; // Compute colour
	eta(i)
	  -= 0.5 * this->spinStructures_[mu](alpha, beta)
	  * this->boundaryConditions_[etaSiteIndex][mu]
	  * conj(this->lattice_->getLink(4 * siteBehindIndex + mu)(b, a))
	  * psi(12 * siteBehindIndex + 3 * beta + b)
	  / this->lattice_->u0();
	
	eta(i)
	  -= 0.5 * this->spinStructures_[mu + 4](alpha, beta)
	  * this->boundaryConditions_[etaSiteIndex][mu + 4]
	  * this->lattice_->getLink(4 * etaSiteIndex + mu)(a, b)
	  * psi(12 * siteAheadIndex + 3 * beta + b)
	  / this->lattice_->u0();
      }
    }
  }

  return eta;
}



VectorXcd UnpreconditionedWilson::applyHermitian(const VectorXcd& psi)
{
  // Right multiply a vector by the operator
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

#pragma omp parallel for
  for (int i = 0; i < this->operatorSize_; ++i) {
    int etaSiteIndex = i / 12; // Site index of the current row in 
    int alpha = (i % 12) / 3; // Spin index of the current row in eta
    int a = i % 3; // Colour index of the current row in eta

    // Now add the mass term
    eta(i) = (1 + 3 / this->lattice_->chi() + this->mass_)
      * pyQCD::gamma5(alpha, alpha) * psi(i);

    // Loop over the four gamma indices (mu) in the sum inside the Wilson action
    for (int mu = 0; mu < 4; ++mu) {
      
      // Now we determine the indices of the neighbouring links

      int siteBehindIndex = this->nearestNeighbours_[etaSiteIndex][mu];
      int siteAheadIndex = this->nearestNeighbours_[etaSiteIndex][mu + 4];

      // Now loop over spin and colour indices for the portion of the row we're
      // on

      for (int j = 0; j < 12; ++j) {
	int beta = j / 3; // Compute spin
	int b = j % 3; // Compute colour
	eta(i)
	  -= 0.5 * this->hermitianSpinStructures_[mu](alpha, beta)
	  * this->boundaryConditions_[etaSiteIndex][mu]
	  * conj(this->lattice_->getLink(4 * siteBehindIndex + mu)(b, a))
	  * psi(12 * siteBehindIndex + 3 * beta + b)
	  / this->lattice_->u0();
	
	eta(i)
	  -= 0.5 * this->hermitianSpinStructures_[mu + 4](alpha, beta)
	  * this->boundaryConditions_[etaSiteIndex][mu + 4]
	  * this->lattice_->getLink(4 * etaSiteIndex + mu)(a, b)
	  * psi(12 * siteAheadIndex + 3 * beta + b)
	  / this->lattice_->u0();
      }
    }
  }

  return eta;
}



VectorXcd UnpreconditionedWilson::undoHermiticity(const VectorXcd& psi)
{
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_);

#pragma omp parallel for
  for (int i = 0; i < this->operatorSize_ / 12; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 4; ++k)
	for (int l = 0; l < 3; ++l)
	  eta(12 * i + 3 * j + l)
	    += pyQCD::gamma5(j, k) * psi(12 * i + 3 * k + l);

  return eta;
}


